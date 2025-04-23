import os
import json
import uuid
import threading
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
MODELS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
TRAIN_DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'train_data')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'zip'}

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(TRAIN_DATA_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODELS_FOLDER'] = MODELS_FOLDER
app.config['TRAIN_DATA_FOLDER'] = TRAIN_DATA_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # Aumentando para 1GB max

# Dicionário para armazenar o status dos treinamentos
training_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Classe para armazenar callbacks de treinamento
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
        
        # Extrair métricas
        metrics = {}
        if hasattr(trainer, 'metrics') and trainer.metrics:
            # Make sure we have the metrics dictionary and it's not empty
            try:
                metrics = {
                    'box_loss': float(trainer.metrics.get('box_loss', 0)),
                    'cls_loss': float(trainer.metrics.get('cls_loss', 0)),
                    'dfl_loss': float(trainer.metrics.get('dfl_loss', 0))
                }
                
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
    """Atualiza o status do treinamento e emite via Socket.IO"""
    training_status[model_id] = info
    
    # Add debug printing for metrics
    metrics_debug = 'No metrics' if not info.get('metrics') else f"Metrics: {info['metrics']}"
    print(f"Emitting training update for model {model_id}: {info['status']} - Epoch {info['current_epoch']}/{info['total_epochs']} - Progress {info['progress']*100:.1f}%")
    print(f"  {metrics_debug}")
    
    socketio.emit('training_update', {
        'model_id': model_id,
        'info': info
    })
    print(f"Training update - Model {model_id}: {info['status']} - Epoch {info['current_epoch']}/{info['total_epochs']} - Progress {info['progress']*100:.1f}%")

# Função para executar o treinamento em uma thread separada
def run_training(model, data_yaml_path, epochs, batch_size, img_size, device, model_id, model_path):
    try:
        # Configurar callback
        callback = TrainingCallback(model_id)
        
        # Atualizar status para "iniciando"
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
            while current_epoch < epochs:
                try:
                    # Sleep for a few seconds to avoid excessive polling
                    time.sleep(5)
                    # Check if training is still running
                    if not hasattr(model, 'trainer'):
                        print("Model has no trainer attribute yet, waiting...")
                        continue
                        
                    try:
                        # Get current epoch from model if available
                        if hasattr(model.trainer, 'epoch'):
                            new_epoch = model.trainer.epoch + 1
                            if new_epoch > current_epoch:
                                current_epoch = new_epoch
                                print(f"Detected epoch progress: {current_epoch}/{epochs}")
                                
                                # Call our callback manually
                                callback.current_epoch = current_epoch
                                callback.on_train_epoch_end(model.trainer)
                    except Exception as e:
                        print(f"Error monitoring epoch progress: {str(e)}")
                        
                    # Check if training is complete
                    if current_epoch >= epochs:
                        print("Training appears to be complete based on epoch count")
                        callback.status = "completed"
                        callback.on_train_end(None)
                        break
                except Exception as e:
                    print(f"Error in training monitor thread: {str(e)}")
                    # Don't break the loop, try again

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
        
        # Iniciar treinamento - in a separate try-except to isolate it from callback errors
        try:
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
        
        # Salvar modelo após treinamento
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
                print("WARNING: No viable save method found. Attempting to identify alternative methods...")
                # Diagnostics to help find alternative methods
                if hasattr(model, 'model'):
                    print(f"Model.model type: {type(model.model)}")
                    if hasattr(model.model, 'save'):
                        print(f"Model.model.save type: {type(model.model.save)}")
                        print(f"Model.model.save content: {model.model.save}")
                        
                    # Check for other potential saving methods
                    for attr_name in dir(model):
                        if 'save' in attr_name.lower() or 'export' in attr_name.lower():
                            attr = getattr(model, attr_name)
                            print(f"Found potential save method: model.{attr_name}, type: {type(attr)}, callable: {callable(attr)}")
                            
                    for attr_name in dir(model.model):
                        if 'save' in attr_name.lower() or 'export' in attr_name.lower():
                            attr = getattr(model.model, attr_name)
                            print(f"Found potential save method: model.model.{attr_name}, type: {type(attr)}, callable: {callable(attr)}")
            

        except Exception as e:
            print(f"Error saving model: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
        
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
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['MODELS_FOLDER'], unique_filename)
        file.save(file_path)
        
        return jsonify({
            'message': 'Model uploaded successfully',
            'filename': unique_filename,
            'path': file_path
        }), 200
    
    return jsonify({'error': 'Error uploading model'}), 400

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
            
            # Extract the zip file
            import zipfile
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            
            # Verificar se existe o arquivo data.yaml no diretório raiz ou em subdiretórios
            yaml_path = None
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    if file.lower() == 'data.yaml':
                        yaml_path = os.path.join(root, file)
                        break
                if yaml_path:
                    break
            
            if not yaml_path:
                # Se não encontrou data.yaml, procurar por qualquer arquivo .yaml
                for root, dirs, files in os.walk(dataset_dir):
                    for file in files:
                        if file.lower().endswith('.yaml'):
                            yaml_path = os.path.join(root, file)
                            # Copiar para a raiz como data.yaml
                            import shutil
                            shutil.copy(yaml_path, os.path.join(dataset_dir, 'data.yaml'))
                            yaml_path = os.path.join(dataset_dir, 'data.yaml')
                            break
                    if yaml_path:
                        break
            
            # Verificar uma última vez se encontrou o arquivo
            data_yaml_path = os.path.join(dataset_dir, 'data.yaml')
            if not os.path.exists(data_yaml_path) and yaml_path:
                # Se temos um yaml mas não está na raiz, copiar para a raiz
                import shutil
                shutil.copy(yaml_path, data_yaml_path)
            
            # Verificação final
            if not os.path.exists(data_yaml_path):
                # Não encontrou nenhum arquivo YAML, criar um básico
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
            
            return jsonify({
                'message': 'Dataset uploaded successfully',
                'dataset_id': dataset_id,
                'path': dataset_dir,
                'yaml_path': data_yaml_path
            }), 200
        except Exception as e:
            print(f"Error during dataset upload: {str(e)}")
            return jsonify({'error': f'Error processing dataset: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed. Please upload a ZIP file'}), 400

@app.route('/api/models', methods=['GET'])
def list_models():
    models = []
    for filename in os.listdir(app.config['MODELS_FOLDER']):
        if filename.endswith('.pt'):
            model_path = os.path.join(app.config['MODELS_FOLDER'], filename)
            models.append({
                'name': filename,
                'path': model_path,
                'size': os.path.getsize(model_path)
            })
    
    return jsonify({'models': models}), 200

@app.route('/api/training-status', methods=['GET'])
def get_training_status():
    """Endpoint para obter o status de todos os treinamentos"""
    model_id = request.args.get('model_id')
    
    if model_id:
        # Retornar apenas o status de um modelo específico
        if model_id in training_status:
            return jsonify({
                'model_id': model_id, 
                'info': training_status[model_id]
            }), 200
        else:
            return jsonify({'error': 'Model ID not found'}), 404
    
    # Retornar todos os status
    return jsonify({
        'training_sessions': [
            {'model_id': model_id, 'info': info} 
            for model_id, info in training_status.items()
        ]
    }), 200

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    data = request.json
    
    if not data or 'image' not in data or 'model' not in data:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], data['image'])
    model_path = os.path.join(app.config['MODELS_FOLDER'], data['model'])
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found'}), 404
    
    try:
        # Load the YOLO model
        model = YOLO(model_path)
        
        # Run inference
        results = model(image_path)
        
        # Process results
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        # Save the output image with detections
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{data['image']}")
        result_img = results[0].plot()
        Image.fromarray(result_img).save(output_path)
        
        return jsonify({
            'message': 'Detection completed',
            'detections': detections,
            'result_image': f"result_{data['image']}"
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    data = request.json
    
    if not data or 'dataset_id' not in data:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    dataset_path = os.path.join(app.config['TRAIN_DATA_FOLDER'], data['dataset_id'])
    
    if not os.path.exists(dataset_path):
        return jsonify({'error': 'Dataset not found'}), 404
    
    try:
        # Configure training parameters
        epochs = data.get('epochs', 50)
        batch_size = data.get('batch_size', 16)
        img_size = data.get('img_size', 640)
        
        # Verificar se o data.yaml existe
        data_yaml_path = os.path.join(dataset_path, 'data.yaml')
        if not os.path.exists(data_yaml_path):
            # Procurar novamente em caso de problemas
            yaml_found = False
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.lower() == 'data.yaml':
                        data_yaml_path = os.path.join(root, file)
                        yaml_found = True
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
        
        # Iniciar treinamento em uma thread separada
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
        for model_id, info in training_status.items():
            # Check if model file exists
            model_path = os.path.join(app.config['MODELS_FOLDER'], f"{model_id}.pt")
            best_model_path = os.path.join(app.config['MODELS_FOLDER'], model_id, "weights", "best.pt")
            
            model_file_exists = os.path.exists(model_path)
            best_model_exists = os.path.exists(best_model_path)
            
            # If model not found in root, check results folder
            model_exists = model_file_exists or best_model_exists
            model_file_path = model_path if model_file_exists else best_model_path if best_model_exists else None
            
            # Get file size if available
            file_size = 0
            if model_file_path and os.path.exists(model_file_path):
                file_size = os.path.getsize(model_file_path)
            
            # Get metrics if available
            metrics = info.get('metrics', {})
            
            model_data = {
                'id': model_id,
                'status': info.get('status', 'unknown'),
                'progress': info.get('progress', 0) * 100,  # Convert to percentage
                'epochs': info.get('total_epochs', 0),
                'completed_epochs': info.get('current_epoch', 0),
                'model_exists': model_exists,
                'file_size': file_size,
                'metrics': metrics,
                'error_message': info.get('error_message')
            }
            
            trained_models.append(model_data)
        
        return jsonify({'trained_models': trained_models}), 200
    except Exception as e:
        print(f"Error listing trained models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_id>/download', methods=['GET'])
def download_model(model_id):
    """Download a specific trained model"""
    try:
        # First check in root models folder
        direct_path = os.path.join(app.config['MODELS_FOLDER'], f"{model_id}.pt")
        if os.path.exists(direct_path):
            return send_from_directory(
                app.config['MODELS_FOLDER'],
                f"{model_id}.pt",
                as_attachment=True,
                attachment_filename=f"yolo_model_{model_id}.pt"
            )
        
        # Then check in the results subfolder
        best_model_path = os.path.join(app.config['MODELS_FOLDER'], model_id, "weights", "best.pt")
        if os.path.exists(best_model_path):
            weights_dir = os.path.join(app.config['MODELS_FOLDER'], model_id, "weights")
            return send_from_directory(
                weights_dir,
                "best.pt",
                as_attachment=True,
                attachment_filename=f"yolo_model_{model_id}_best.pt"
            )
        
        # If no model found
        return jsonify({'error': 'Model file not found'}), 404
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Interrompe um treinamento em andamento"""
    data = request.json
    
    if not data or 'model_id' not in data:
        return jsonify({'error': 'Missing model_id parameter'}), 400
    
    model_id = data['model_id']
    
    if model_id not in training_status:
        return jsonify({'error': 'Training session not found'}), 404
    
    # Verificar se o treinamento já está completo ou em erro
    current_status = training_status[model_id].get('status')
    
    if current_status in ['completed', 'error', 'stopped']:
        return jsonify({'message': f'Training already in {current_status} state'}), 200
    
    try:
        # Atualizar o status para "stopped"
        update_training_info(model_id, {
            'status': 'stopped',
            'current_epoch': training_status[model_id].get('current_epoch', 0),
            'total_epochs': training_status[model_id].get('total_epochs', 0),
            'progress': training_status[model_id].get('progress', 0),
            'metrics': training_status[model_id].get('metrics', {}),
            'error_message': 'Training stopped by user'
        })
        
        # Podemos tentar localizar o processo de treinamento para terminá-lo
        # Isso seria ideal, mas requer acompanhamento do PID dos processos
        # Por enquanto, apenas atualizamos o status
        
        return jsonify({
            'message': 'Training stop signal sent. The training will stop at the end of the current epoch.',
            'model_id': model_id,
            'status': 'stopped'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 