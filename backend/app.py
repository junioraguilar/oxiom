import os
import json
import uuid
import threading
import time
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from models import db, YoloModel, Dataset
import models
import shutil
import psutil

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
MODELS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
TRAIN_DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'train_data')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'zip', 'pt'}
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_db.sqlite')

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(TRAIN_DATA_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODELS_FOLDER'] = MODELS_FOLDER
app.config['TRAIN_DATA_FOLDER'] = TRAIN_DATA_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # Increase to 1GB max
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DATABASE_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db.init_app(app)

# Create tables if they don't exist
with app.app_context():
    db.create_all()
    # Fix status of models "in progress" after restart
    models_in_progress = YoloModel.query.filter(YoloModel.status.in_(['training', 'starting', 'running'])).all()
    for model in models_in_progress:
        model.status = 'error'  # or 'stopped' if preferred
        model.error_message = 'Training interrupted by backend restart.'
        db.session.commit()

# Dictionary to store training status
training_status = {}
# Dictionary for training stop flags
training_stop_flags = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Class to store training callbacks
class TrainingCallback:
    def __init__(self, model_id):
        self.model_id = model_id
        self.current_epoch = 0
        self.total_epochs = 0
        self.metrics = {}
        self.status = "preparing"
        
    def on_train_start(self, trainer):
        self.status = "running"
        self.total_epochs = trainer.epochs
        update_training_info(self.model_id, {
            'status': self.status,
            'current_epoch': 0,
            'total_epochs': self.total_epochs,
            'progress': 0,
            'metrics': {}
        })
    
    def on_train_epoch_end(self, trainer):
        self.current_epoch = trainer.epoch + 1
        # Calculate progress as a decimal (0.0 to 1.0)
        progress = float(self.current_epoch) / float(self.total_epochs)
        
        # Extract metrics
        metrics = {}
        if hasattr(trainer, 'metrics') and trainer.metrics:
            try:
                # Try to extract validation metrics if available
                if 'metrics/precision(B)' in trainer.metrics:
                    metrics['precision'] = float(trainer.metrics.get('metrics/precision(B)', 0))
                    metrics['recall'] = float(trainer.metrics.get('metrics/recall(B)', 0))
                    metrics['map50'] = float(trainer.metrics.get('metrics/mAP50(B)', 0))
                    metrics['map'] = float(trainer.metrics.get('metrics/mAP50-95(B)', 0))
                
                print(f"Extracted metrics: {metrics}")
            except Exception as e:
                print(f"Error extracting metrics: {str(e)}")
                print(f"Trainer metrics: {trainer.metrics}")
        
        # Add GPU_mem (in GB)
        gpu_mem = 0.0
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem = meminfo.used / (1024 ** 3)  # in GB
            pynvml.nvmlShutdown()
        except Exception as e:
            # Fallback to torch if pynvml is not available
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_reserved() / (1024 ** 3)
            except Exception:
                gpu_mem = 0.0
        metrics['GPU_mem'] = round(gpu_mem, 3)
        
        # Real-time epoch percent (simulate as 100% at epoch end)
        metrics['epoch_percent'] = 100.0
        
        self.metrics = metrics
        
        update_training_info(self.model_id, {
            'status': self.status,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'progress': progress,
            'metrics': metrics
        })
    
    def on_train_end(self, trainer):
        # Set status first, regardless of what happens next
        self.status = "completed"
        
        # Special case for manual calling with None
        if trainer is None:
            print("Manual on_train_end call with None trainer - using stored metrics")
            update_training_info(self.model_id, {
                'status': self.status,
                'current_epoch': self.total_epochs,
                'total_epochs': self.total_epochs,
                'progress': 1.0,
                'metrics': self.metrics
            })
            return
        
        # Handle any type of trainer object safely
        try:
            # Check if trainer is a valid object
            if not hasattr(trainer, 'epoch'):
                print("Trainer object has no 'epoch' attribute, using stored metrics")
            elif isinstance(trainer, (list, tuple)):
                print("Trainer is a list or tuple, cannot use directly")
            elif callable(trainer):
                print("Trainer appears to be a function or callable, not an object")
            else:
                # Trainer looks valid, but let's check its attributes safely
                try:
                    # Try to access some attributes to see if they're valid
                    epoch = getattr(trainer, 'epoch', self.total_epochs - 1)
                    self.current_epoch = epoch + 1
                    print(f"Using trainer.epoch: {epoch}")
                except Exception as e:
                    print(f"Error accessing trainer.epoch: {str(e)}")
        except Exception as e:
            print(f"Error in on_train_end type checking: {str(e)}")
        
        # Always update with stored values for consistency
        update_training_info(self.model_id, {
            'status': self.status,
            'current_epoch': self.total_epochs,
            'total_epochs': self.total_epochs,
            'progress': 1.0,
            'metrics': self.metrics
        })

def update_training_info(model_id, info):
    """Update training status and emit via Socket.IO"""
    training_status[model_id] = info
    
    # Add debug printing for metrics
    metrics_debug = 'No metrics' if not info.get('metrics') else f"Metrics: {info['metrics']}"
    print(f"Emitting training update for model {model_id}: {info['status']} - Epoch {info['current_epoch']}/{info['total_epochs']} - Progress {info['progress']*100:.1f}%")
    print(f"  {metrics_debug}")
    
    # Update the database
    with app.app_context():
        model = YoloModel.query.get(model_id)
        if model:
            model.status = info['status']
            model.current_epoch = info['current_epoch']
            model.total_epochs = info['total_epochs']
            model.progress = info['progress']
            if 'metrics' in info:
                model.metrics = info['metrics']
            if 'error_message' in info:
                model.error_message = info['error_message']
            
            # Check if the model file exists and update its size
            model_path = os.path.join(app.config['MODELS_FOLDER'], f"{model_id}.pt")
            best_model_path = os.path.join(app.config['MODELS_FOLDER'], model_id, "weights", "best.pt")
            
            if os.path.exists(model_path):
                model.file_size = os.path.getsize(model_path)
            elif os.path.exists(best_model_path):
                model.file_size = os.path.getsize(best_model_path)
                
            db.session.commit()
        else:
            print(f"Warning: Model {model_id} not found in database - unable to update")
    
    socketio.emit('training_update', {
        'model_id': model_id,
        'info': info
    })
    print(f"Training update - Model {model_id}: {info['status']} - Epoch {info['current_epoch']}/{info['total_epochs']} - Progress {info['progress']*100:.1f}%")

# Function to run training in a separate thread
def run_training(model, data_yaml_path, epochs, batch_size, img_size, device, model_id, model_path):
    try:
        # Configure callback
        callback = TrainingCallback(model_id)
        # Initialize stop flag
        training_stop_flags[model_id] = False
        # Update status to "starting"
        update_training_info(model_id, {
            'status': 'starting',
            'current_epoch': 0,
            'total_epochs': epochs,
            'progress': 0,
            'metrics': {},
            'device': device,
            'yaml_path': data_yaml_path
        })
        # Try an alternative approach for handling callbacks
        # Some versions of Ultralytics may use a completely different method
        print("Using an alternative approach to monitor training instead of direct callbacks")
        # Create a thread to monitor training progress
        def monitor_training_progress():
            current_epoch = 0
            last_epoch_batches = 0
            while current_epoch < epochs:
                # --- NEW: Check if it needs to stop ---
                if training_stop_flags.get(model_id):
                    print("Training stopped by user request (flag detected in monitor thread)")
                    callback.status = "stopped"
                    callback.on_train_end(None)
                    break
                try:
                    # Sleep for 0.1 seconds for real-time updates
                    time.sleep(0.1)
                    if not hasattr(model, 'trainer'):
                        print("Model has no trainer attribute yet, waiting...")
                        continue
                    try:
                        # Get progress within the epoch
                        if hasattr(model.trainer, 'epoch') and hasattr(model.trainer, 'dataloader'):
                            new_epoch = model.trainer.epoch + 1
                            dataloader = getattr(model.trainer, 'dataloader', None)
                            total_batches = len(dataloader) if dataloader is not None else 0
                            # Try different attributes for the current batch
                            batch_idx = None
                            for attr in ['batch_i', 'batch', 'iter', 'iteration', 'batch_idx']:
                                if hasattr(model.trainer, attr):
                                    batch_idx = getattr(model.trainer, attr)
                                    break
                            if batch_idx is None:
                                batch_idx = 0
                            epoch_percent = float(batch_idx + 1) / float(total_batches) * 100 if total_batches else 0.0
                            print(f"[DEBUG] Epoch: {new_epoch}, Batch: {batch_idx}/{total_batches}, Percent: {epoch_percent:.1f}")
                            # Update epoch_percent in real-time
                            info = training_status.get(model_id, {}).copy()
                            if 'metrics' not in info:
                                info['metrics'] = {}
                            info['metrics']['epoch_percent'] = round(epoch_percent, 1)
                            # Update GPU_mem in real-time
                            gpu_mem = 0.0
                            try:
                                import pynvml
                                pynvml.nvmlInit()
                                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                                gpu_mem = meminfo.used / (1024 ** 3)  # in GB
                                pynvml.nvmlShutdown()
                            except Exception as e:
                                # Fallback to torch if pynvml is not available
                                try:
                                    import torch
                                    if torch.cuda.is_available():
                                        gpu_mem = torch.cuda.memory_reserved() / (1024 ** 3)
                                except Exception:
                                    gpu_mem = 0.0
                            info['metrics']['GPU_mem'] = round(gpu_mem, 3)
                            update_training_info(model_id, info)
                            if new_epoch > current_epoch:
                                current_epoch = new_epoch
                                print(f"Detected epoch progress: {current_epoch}/{epochs}")
                                callback.current_epoch = current_epoch
                                callback.on_train_epoch_end(model.trainer)
                    except Exception as e:
                        print(f"Error monitoring epoch progress: {str(e)}")
                    if current_epoch >= epochs:
                        print("Training appears to be complete based on epoch count")
                        callback.status = "completed"
                        callback.on_train_end(None)
                        break
                except Exception as e:
                    print(f"Error in training monitor thread: {str(e)}")
        # Start the monitor as a separate thread
        monitor_thread = None
        try:
            if hasattr(model, 'add_callback'):
                # If add_callback exists, we'll try it first
                print("Registering callbacks using model.add_callback")
                # Wrap each callback registration in its own try-except
                try:
                    model.add_callback('on_train_start', callback.on_train_start)
                except Exception as e:
                    print(f"Error registering on_train_start callback: {str(e)}")
                try:
                    model.add_callback('on_train_epoch_end', callback.on_train_epoch_end)
                except Exception as e:
                    print(f"Error registering on_train_epoch_end callback: {str(e)}")
                try:
                    model.add_callback('on_train_end', callback.on_train_end)
                except Exception as e:
                    print(f"Error registering on_train_end callback: {str(e)}")
            else:
                # Otherwise use our monitoring thread approach
                print("Model doesn't support add_callback, using monitoring thread")
                monitor_thread = threading.Thread(target=monitor_training_progress, daemon=True)
                monitor_thread.start()
        except Exception as e:
            print(f"Error setting up training monitoring: {str(e)}")
            # If all fails, we'll rely on manual updates at the end
        
        # Start training - instead of a manual loop, call once with correct epochs
        try:
            if training_stop_flags.get(model_id):
                print("Training stopped by user request before starting train().")
                callback.status = "stopped"
                callback.on_train_end(None)
            else:
                results = model.train(
                    data=data_yaml_path,
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=img_size,
                    device=device,
                    project=app.config['MODELS_FOLDER'],
                    name=model_id,
                    exist_ok=True
                )
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            raise  # Re-raise to handle in outer try-except
            
        # Manually set the completed status
        print("Training completed, updating final status")
        
        # Safely call on_train_end in its own try-except block
        try:
            callback.status = "completed"
            callback.on_train_end(None)  # Pass None so our safe handler is used
        except Exception as e:
            print(f"Error calling on_train_end: {str(e)}")
            # Manually update the training status if the callback fails
            update_training_info(model_id, {
                'status': 'completed',
                'current_epoch': epochs,
                'total_epochs': epochs,
                'progress': 1.0,
                'metrics': callback.metrics if hasattr(callback, 'metrics') else {}
            })
        
        # Save the model after training
        model_saved = False
        try:
            print("Attempting to save the model...")
            print(f"Model type: {type(model)}")
            
            # For ultralytics 8.3.x, model.save() is the recommended approach
            if hasattr(model, 'save') and callable(getattr(model, 'save')):
                print("Using model.save() - recommended for ultralytics 8.3.x")
                model.save(model_path)
                print(f"Model successfully saved to {model_path}")
                model_saved = True
            # For ultralytics 8.0.x, model.export() might be used
            elif hasattr(model, 'export') and callable(getattr(model, 'export')):
                print("Using model.export() as fallback")
                model.export(model_path)
                print(f"Model successfully exported to {model_path}")
            # Legacy approach for older versions
            elif hasattr(model, 'model') and hasattr(model.model, 'save') and callable(getattr(model.model, 'save')):
                print("Using legacy model.model.save() method")
                model.model.save(model_path)
                print(f"Model successfully saved with legacy method to {model_path}")
                model_saved = True
            
            print(f"Model saved: {model_saved}")
            if not model_saved:
                print("Warning: Model may not have been saved correctly!")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"Error in training thread: {error_msg}")
        update_training_info(model_id, {
            'status': 'error',
            'error_message': error_msg,
            'current_epoch': callback.current_epoch if hasattr(callback, 'current_epoch') else 0,
            'total_epochs': epochs,
            'progress': 0
        })
        return False
    finally:
        # End of the main try block in run_training
        return True

@app.route('/api/upload-image', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': unique_filename,
            'path': file_path
        }), 200
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/upload-model', methods=['POST'])
def upload_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    model_name = request.form.get('name', 'Uploaded model')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique ID for the model
        model_id = str(uuid.uuid4())
        filename = f"{model_id}_" + secure_filename(file.filename)
        file_path = os.path.join(app.config['MODELS_FOLDER'], filename)
        
        # Save the file
        file.save(file_path)
        
        # Get the file size
        file_size = os.path.getsize(file_path)
        
        # Create a record in the database
        with app.app_context():
            upload_model = YoloModel(
                id=model_id,
                name=model_name,
                file_path=filename,
                status='completed',
                progress=1.0,
                file_size=file_size,
                total_epochs=0,
                current_epoch=0
            )
            db.session.add(upload_model)
            db.session.commit()
        
        return jsonify({
            'message': 'Model uploaded successfully',
            'model_id': model_id,
            'filename': filename
        }), 201
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # For dataset, typically a zip file
    if file and file.filename.endswith('.zip'):
        try:
            dataset_id = str(uuid.uuid4())
            dataset_dir = os.path.join(app.config['TRAIN_DATA_FOLDER'], dataset_id)
            os.makedirs(dataset_dir, exist_ok=True)
            
            filename = secure_filename(file.filename)
            file_path = os.path.join(dataset_dir, filename)
            file.save(file_path)
            
            print(f"Dataset saved to {file_path}, size: {os.path.getsize(file_path)} bytes")
            file_size = os.path.getsize(file_path)
            
            # Extract the zip file
            import zipfile
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            
            # Check if there is a data.yaml file in the root directory or subdirectories
            yaml_path = None
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    if file.lower() == 'data.yaml':
                        yaml_path = os.path.join(root, file)
                        break
                if yaml_path:
                    break
            
            if not yaml_path:
                # If no data.yaml is found, look for any .yaml file
                for root, dirs, files in os.walk(dataset_dir):
                    for file in files:
                        if file.lower().endswith('.yaml'):
                            yaml_path = os.path.join(root, file)
                            # Copy to the root as data.yaml
                            import shutil
                            shutil.copy(yaml_path, os.path.join(dataset_dir, 'data.yaml'))
                            yaml_path = os.path.join(dataset_dir, 'data.yaml')
                            break
                    if yaml_path:
                        break
            
            # Final check if a YAML file was found
            data_yaml_path = os.path.join(dataset_dir, 'data.yaml')
            if not os.path.exists(data_yaml_path) and yaml_path:
                # If we have a YAML but it's not in the root, copy it to the root
                import shutil
                shutil.copy(yaml_path, data_yaml_path)
            
            # Final check
            if not os.path.exists(data_yaml_path):
                # No YAML file found, create a basic one
                print("No YAML file found, creating a default one")
                with open(data_yaml_path, 'w') as f:
                    f.write("""
# YOLO Dataset Configuration
path: .
train: images/train
val: images/val
test: images/test

# Classes
nc: 1  # number of classes
names: ['object']  # class names
""")
            
            print(f"Data YAML path: {data_yaml_path}")
            
            # Extract class names from the YAML file
            classes = []
            try:
                import yaml
                with open(data_yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    if 'names' in yaml_data:
                        classes = yaml_data['names']
            except Exception as e:
                print(f"Error extracting classes from YAML: {str(e)}")
            
            # Save to the database
            # Extract the dataset name from the file name (remove .zip)
            dataset_name = os.path.splitext(filename)[0]
            
            with app.app_context():
                dataset = Dataset(
                    id=dataset_id,
                    name=dataset_name,
                    file_path=os.path.relpath(file_path, app.config['TRAIN_DATA_FOLDER']),
                    yaml_path=os.path.relpath(data_yaml_path, app.config['TRAIN_DATA_FOLDER']),
                    file_size=file_size,
                )
                dataset.classes = classes
                
                db.session.add(dataset)
                db.session.commit()
            
            return jsonify({
                'message': 'Dataset uploaded successfully',
                'dataset_id': dataset_id,
                'name': dataset_name,
                'path': dataset_dir,
                'yaml_path': data_yaml_path,
                'classes': classes
            }), 200
        except Exception as e:
            print(f"Error during dataset upload: {str(e)}")
            return jsonify({'error': f'Error processing dataset: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed. Please upload a ZIP file'}), 400

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """List all available datasets"""
    try:
        with app.app_context():
            db_datasets = Dataset.query.order_by(Dataset.created_at.desc()).all()
            
            datasets = []
            for dataset in db_datasets:
                datasets.append(dataset.to_dict())
            
            return jsonify({'datasets': datasets}), 200
            
    except Exception as e:
        print(f"Error listing datasets: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def list_models():
    """List all available models from the database"""
    try:
        with app.app_context():
            db_models = YoloModel.query.filter(
                YoloModel.status.in_(['completed', 'stopped'])
            ).order_by(YoloModel.created_at.desc()).all()
            
            models = []
            for model in db_models:
                # Check if the model file exists in the file system
                model_path = None
                model_exists = False
                
                # Check possible locations for the model file
                if model.file_path:
                    full_path = os.path.join(app.config['MODELS_FOLDER'], model.file_path)
                    if os.path.exists(full_path):
                        model_path = full_path
                        model_exists = True
                
                if not model_exists:
                    # Check with the direct ID name
                    direct_path = os.path.join(app.config['MODELS_FOLDER'], f"{model.id}.pt")
                    if os.path.exists(direct_path):
                        model_path = direct_path
                        model_exists = True
                
                if not model_exists:
                    # Check in the training directory
                    best_path = os.path.join(app.config['MODELS_FOLDER'], model.id, "weights", "best.pt")
                    if os.path.exists(best_path):
                        model_path = best_path
                        model_exists = True
                
                if model_exists:
                    models.append({
                        'id': model.id,
                        'name': model.name or f"Model {model.id[:8]}",
                        'path': model_path,
                        'size': model.file_size or os.path.getsize(model_path),
                        'created_at': model.created_at.isoformat() if model.created_at else None,
                        'classes': model.classes
                    })
            
            return jsonify({'models': models}), 200
            
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training-status', methods=['GET'])
def get_training_status():
    """Return the status of all ongoing trainings in the format expected by the frontend"""
    try:
        model_id = request.args.get('model_id')
        if model_id:
            # Individual query (maintains compatibility)
            with app.app_context():
                model = YoloModel.query.get(model_id)
                if not model:
                    return jsonify({'error': 'Model not found'}), 404
                current_info = training_status.get(model_id)
                if current_info:
                    info = {
                        'status': current_info.get('status', model.status),
                        'progress': current_info.get('progress', model.progress) * 100,
                        'current_epoch': current_info.get('current_epoch', model.current_epoch),
                        'total_epochs': current_info.get('total_epochs', model.total_epochs),
                        'metrics': current_info.get('metrics', model.metrics),
                        'error_message': current_info.get('error_message', model.error_message)
                    }
                else:
                    info = model.to_dict()
                return jsonify({'training_sessions': [{ 'model_id': model_id, 'info': info }]})
        else:
            # Query all ongoing models
            with app.app_context():
                training_models = YoloModel.query.filter(
                    YoloModel.status.in_(['training', 'starting', 'running'])
                ).all()
                sessions = []
                for model in training_models:
                    current_info = training_status.get(model.id)
                    if current_info:
                        info = {
                            'status': current_info.get('status', model.status),
                            'progress': current_info.get('progress', model.progress) * 100,
                            'current_epoch': current_info.get('current_epoch', model.current_epoch),
                            'total_epochs': current_info.get('total_epochs', model.total_epochs),
                            'metrics': current_info.get('metrics', model.metrics),
                            'error_message': current_info.get('error_message', model.error_message)
                        }
                    else:
                        info = model.to_dict()
                    sessions.append({'model_id': model.id, 'info': info})
                return jsonify({'training_sessions': sessions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    model_id = request.form.get('model_id')
    if not model_id:
        return jsonify({'error': 'No model specified'}), 400
    
    try:
        # Find the model in the database
        with app.app_context():
            model_record = YoloModel.query.get(model_id)
            if not model_record:
                return jsonify({'error': f'Model with ID {model_id} not found in database'}), 404
        
        # Check possible locations for the model file
        model_path = os.path.join(app.config['MODELS_FOLDER'], f"{model_id}.pt")
        if not os.path.exists(model_path):
            # Check if it's a model uploaded with a custom name
            if model_record.file_path and model_record.file_path != f"{model_id}.pt":
                alt_model_path = os.path.join(app.config['MODELS_FOLDER'], model_record.file_path)
                if os.path.exists(alt_model_path):
                    model_path = alt_model_path
                else:
                    # Check in the training directory
                    best_path = os.path.join(app.config['MODELS_FOLDER'], model_id, "weights", "best.pt")
                    if os.path.exists(best_path):
                        model_path = best_path
                    else:
                        return jsonify({'error': 'Model file not found on disk'}), 404
        
        file = request.files['file']
        confidence = float(request.form.get('confidence', 0.25))
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Detect objects
            model = YOLO(model_path)
            results = model(file_path, conf=confidence)
            result = results[0]  # Just take the first result
            
            # Process results
            detections = []
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                
                # Get coordinates
                x1, y1, x2, y2 = [float(coord) for coord in box.xyxy[0].tolist()]
                
                # Map class ID to label
                classes = model_record.classes if model_record.classes else []
                class_name = classes[class_id] if class_id < len(classes) else f"Class {class_id}"
                
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'box': {
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'width': x2 - x1,
                        'height': y2 - y1
                    }
                })
            
            # Load image for dimensions
            img = Image.open(file_path)
            width, height = img.size
            
            return jsonify({
                'detections': detections,
                'image_path': f"/api/uploads/{filename}",
                'image_dimensions': {
                    'width': width,
                    'height': height
                }
            }), 200
        
        return jsonify({'error': 'File type not allowed'}), 400
        
    except Exception as e:
        print(f"Detection error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    data = request.json
    
    if not data or 'dataset_id' not in data:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        # Get the dataset from the database
        with app.app_context():
            dataset = Dataset.query.get(data['dataset_id'])
            if not dataset:
                return jsonify({'error': 'Dataset not found in database'}), 404
        
        # Dataset path from the uploads directory
        dataset_path = os.path.join(app.config['TRAIN_DATA_FOLDER'], dataset.id)
    
        if not os.path.exists(dataset_path):
                return jsonify({'error': 'Dataset files not found on disk'}), 404
    
        # Configure training parameters
        epochs = data.get('epochs', 50)
        batch_size = data.get('batch_size', 16)
        img_size = data.get('img_size', 640)
        model_name = data.get('name', f"Model from {dataset.name}")
        
        # Use the yaml_path from the database or verify its existence
        data_yaml_path = os.path.join(dataset_path, 'data.yaml')
        if dataset.yaml_path:
            full_yaml_path = os.path.join(app.config['TRAIN_DATA_FOLDER'], dataset.yaml_path)
            if os.path.exists(full_yaml_path):
                data_yaml_path = full_yaml_path
        
        # Double-check that the YAML exists
        if not os.path.exists(data_yaml_path):
            # Look again in case of issues
            yaml_found = False
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.lower() == 'data.yaml':
                        data_yaml_path = os.path.join(root, file)
                        break
                if yaml_found:
                    break
            
            if not yaml_found:
                return jsonify({'error': 'data.yaml not found in dataset'}), 400
        
        print(f"Using data YAML: {data_yaml_path}")
        
        # Initialize a new YOLO model - use YOLOv8n instead of YOLOv11
        model = YOLO('yolov8n.pt')  # Start with pre-trained YOLOv8 nano model
        
        # Check if CUDA is available
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Start training process
        model_id = str(uuid.uuid4())
        model_path = os.path.join(app.config['MODELS_FOLDER'], f"{model_id}.pt")
        
        # Use the classes from the dataset or read them again from the YAML
        classes = dataset.classes
        if not classes:
            try:
                import yaml
                with open(data_yaml_path, 'r') as yaml_file:
                    yaml_data = yaml.safe_load(yaml_file)
                    classes = yaml_data.get('names', [])
            except Exception as e:
                print(f"Failed to read classes from YAML: {str(e)}")
        
        # Create a record in the database
        with app.app_context():
            training_model = YoloModel(
                id=model_id,
                name=model_name,
                file_path=f"{model_id}.pt",
                dataset_id=data['dataset_id'],
                status='starting',
                total_epochs=epochs,
                current_epoch=0,
                progress=0.0,
                parameters={
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'img_size': img_size,
                    'device': device
                },
                classes=classes
            )
            db.session.add(training_model)
            db.session.commit()
        
        # Notify the frontend immediately about the new training
        update_training_info(model_id, {
            'status': 'starting',
            'current_epoch': 0,
            'total_epochs': epochs,
            'progress': 0,
            'metrics': {},
            'device': device,
            'yaml_path': data_yaml_path
        })
        
        # Start training in a separate thread
        training_thread = threading.Thread(
            target=run_training,
            args=(model, data_yaml_path, epochs, batch_size, img_size, device, model_id, model_path),
            daemon=True
        )
        training_thread.start()
        
        return jsonify({
            'message': 'Training job started',
            'model_id': model_id,
            'status': 'starting',
            'device': device,
            'yaml_path': data_yaml_path
        }), 200
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/uploads/<filename>', methods=['GET'])
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@app.route('/api/trained-models', methods=['GET'])
def list_trained_models():
    """List trained models with their details"""
    trained_models = []
    
    try:
        # Find models in the database
        with app.app_context():
            models = YoloModel.query.order_by(YoloModel.created_at.desc()).all()
            
            for model in models:
                # Check if the model file exists in the file system
                model_path = os.path.join(app.config['MODELS_FOLDER'], f"{model.id}.pt")
                best_model_path = os.path.join(app.config['MODELS_FOLDER'], model.id, "weights", "best.pt")
                
                model_file_exists = os.path.exists(model_path)
                best_model_exists = os.path.exists(best_model_path)
                model_exists = model_file_exists or best_model_exists
                
                # Update the file size if it exists
                if model_file_exists:
                    file_size = os.path.getsize(model_path)
                    if model.file_size != file_size:
                        model.file_size = file_size
                        db.session.commit()
                elif best_model_exists:
                    file_size = os.path.getsize(best_model_path)
                    if model.file_size != file_size:
                        model.file_size = file_size
                        db.session.commit()
                
                # Convert the model to a dictionary and add it to the list
                model_dict = model.to_dict()
                model_dict['model_exists'] = model_exists
                trained_models.append(model_dict)
        
        return jsonify({'trained_models': trained_models}), 200
    except Exception as e:
        print(f"Error listing trained models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_id>/download', methods=['GET'])
def download_model(model_id):
    """Download a specific trained model"""
    try:
        # Check if the model exists in the database
        with app.app_context():
            model = YoloModel.query.get(model_id)
            if not model:
                return jsonify({'error': 'Model not found in database'}), 404
        
        # Check in the file system
        direct_path = os.path.join(app.config['MODELS_FOLDER'], f"{model_id}.pt")
        if os.path.exists(direct_path):
            return send_from_directory(
                app.config['MODELS_FOLDER'],
                f"{model_id}.pt",
                as_attachment=True,
                attachment_filename=f"yolo_model_{model_id}.pt"
            )
        
        # Check if it's a model uploaded with a custom name
        if model.file_path and model.file_path != f"{model_id}.pt":
            model_file_path = os.path.join(app.config['MODELS_FOLDER'], model.file_path)
            if os.path.exists(model_file_path):
                filename = os.path.basename(model_file_path)
                return send_from_directory(
                    app.config['MODELS_FOLDER'],
                    filename,
                    as_attachment=True,
                    attachment_filename=f"yolo_model_{model_id}_{filename.split('_', 1)[1] if '_' in filename else filename}"
                )
        
        # Check in the training directory
        best_model_path = os.path.join(app.config['MODELS_FOLDER'], model_id, "weights", "best.pt")
        if os.path.exists(best_model_path):
            weights_dir = os.path.join(app.config['MODELS_FOLDER'], model_id, "weights")
            return send_from_directory(
                weights_dir,
                "best.pt",
                as_attachment=True,
                attachment_filename=f"yolo_model_{model_id}_best.pt"
            )
        
        # No file found
        return jsonify({'error': 'Model file not found in filesystem'}), 404
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop an ongoing training"""
    data = request.json
    
    if not data or 'model_id' not in data:
        return jsonify({'error': 'Missing model_id parameter'}), 400
    
    model_id = data['model_id']
    
    # Check if the model exists in the database
    with app.app_context():
        model = YoloModel.query.get(model_id)
        if not model:
            return jsonify({'error': 'Training session not found'}), 404
        
        # Check if the training is already complete or in error
        if model.status in ['completed', 'error', 'stopped']:
            return jsonify({'message': f'Training already in {model.status} state'}), 200
        
        # Update the model status in the database
        model.status = 'stopped'
        model.error_message = 'Training stopped by user'
        db.session.commit()
    
    # Update the in-memory status if it exists
    if model_id in training_status:
        update_training_info(model_id, {
            'status': 'stopped',
            'current_epoch': training_status[model_id].get('current_epoch', 0),
            'total_epochs': training_status[model_id].get('total_epochs', 0),
            'progress': training_status[model_id].get('progress', 0),
            'metrics': training_status[model_id].get('metrics', {}),
            'error_message': 'Training stopped by user'
        })
    
    # Signal stop to the training thread
    training_stop_flags[model_id] = True
    
    return jsonify({
        'message': 'Training stop signal sent. The training will stop at the next check.',
        'model_id': model_id,
        'status': 'stopped'
    }), 200

@app.route('/api/models/<model_id>/delete', methods=['DELETE', 'OPTIONS'])
def delete_model(model_id):
    """Delete a trained model from the database and file system"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Check if the model exists in the database
        with app.app_context():
            model = YoloModel.query.get(model_id)
            if not model:
                return jsonify({'error': 'Model not found in database'}), 404
            
            # Delete the files before removing from the database
            deleted_files = []

            # Check if there is a direct file with the model ID
            direct_path = os.path.join(app.config['MODELS_FOLDER'], f"{model_id}.pt")
            if os.path.exists(direct_path):
                os.remove(direct_path)
                deleted_files.append(direct_path)
            
            # Check if there is a file with a custom name
            if model.file_path and model.file_path != f"{model_id}.pt":
                custom_path = os.path.join(app.config['MODELS_FOLDER'], model.file_path)
                if os.path.exists(custom_path):
                    os.remove(custom_path)
                    deleted_files.append(custom_path)
            
            # Check if there is a training directory
            results_dir = os.path.join(app.config['MODELS_FOLDER'], model_id)
            if os.path.exists(results_dir) and os.path.isdir(results_dir):
                import shutil
                shutil.rmtree(results_dir)
                deleted_files.append(results_dir)
            
            # Remove from the database
            db.session.delete(model)
            db.session.commit()
            
            # Remove from the in-memory status if it exists
            if model_id in training_status:
                del training_status[model_id]
            
            return jsonify({
                'message': 'Model deleted successfully',
                'model_id': model_id,
                'deleted_files': deleted_files
            }), 200
            
    except Exception as e:
        print(f"Error deleting model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/datasets/<dataset_id>/delete', methods=['DELETE', 'OPTIONS'])
def delete_dataset(dataset_id):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        with app.app_context():
            dataset = Dataset.query.get(dataset_id)
            if not dataset:
                return jsonify({'error': 'Dataset not found'}), 404
            # Remove dataset directory
            dataset_dir = os.path.join(app.config['TRAIN_DATA_FOLDER'], dataset_id)
            if os.path.exists(dataset_dir):
                shutil.rmtree(dataset_dir)
            # Delete from database
            db.session.delete(dataset)
            db.session.commit()
        return jsonify({'message': 'Dataset deleted successfully'}), 200
    except Exception as e:
        print(f"Error deleting dataset {dataset_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/datasets/<dataset_id>/download', methods=['GET', 'OPTIONS'])
def download_dataset_file(dataset_id):
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
    # Retrieve dataset record
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    # Serve the stored file
    file_abs = os.path.join(app.config['TRAIN_DATA_FOLDER'], dataset.file_path)
    print(f"Downloading dataset {dataset_id} from {file_abs}")
    if not os.path.exists(file_abs):
        return jsonify({'error': 'File not found'}), 404
    return send_file(file_abs, as_attachment=True)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 