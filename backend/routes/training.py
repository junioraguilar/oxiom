from flask import Blueprint, request, jsonify
import os
import torch
import uuid
from extensions import db, socketio
from config import Config
from models import YoloModel, Dataset
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import threading

training_bp = Blueprint('training', __name__)

# In-memory stores for status/flags (should be improved for production)
training_status = {}
training_stop_flags = {}

def update_training_info(model_id, info):
    training_status[model_id] = info
    print(f"[TRAINING] Updated training info for model_id={model_id}: {info}")
    socketio.emit('training_update', {'model_id': model_id, 'info': info})

@training_bp.route('/api/train', methods=['POST'])
def train_model():
    data = request.json
    if not data or 'dataset_id' not in data:
        return jsonify({'error': 'Missing required parameters'}), 400
    try:
        dataset = Dataset.query.get(data['dataset_id'])
        if not dataset:
            return jsonify({'error': 'Dataset not found in database'}), 404
        dataset_path = os.path.join(Config.TRAIN_DATA_FOLDER, dataset.id)
        if not os.path.exists(dataset_path):
            return jsonify({'error': 'Dataset files not found on disk'}), 404
        epochs = data.get('epochs', 50)
        batch_size = data.get('batch_size', 16)
        img_size = data.get('img_size', 640)
        model_name = data.get('name', f"Model from {dataset.name}")
        model_type = data.get('model_type', 'n').lower()
        model_types_map = {'n': 'yolov8n.pt','s': 'yolov8s.pt','m': 'yolov8m.pt','l': 'yolov8l.pt','x': 'yolov8x.pt'}
        if model_type not in model_types_map:
            return jsonify({'error': f"Invalid model_type '{model_type}'. Choose one of: n, s, m, l, x."}), 400
        model_pt = model_types_map[model_type]
        data_yaml_path = os.path.join(dataset_path, 'data.yaml')
        if dataset.yaml_path:
            full_yaml_path = os.path.join(Config.TRAIN_DATA_FOLDER, dataset.yaml_path)
            if os.path.exists(full_yaml_path):
                data_yaml_path = full_yaml_path
        if not os.path.exists(data_yaml_path):
            return jsonify({'error': 'data.yaml not found in dataset'}), 400
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model_id = str(uuid.uuid4())
        model_path = os.path.join(Config.MODELS_FOLDER, f"{model_id}.pt")
        classes = dataset.classes
        if not classes:
            try:
                import yaml
                with open(data_yaml_path, 'r') as yaml_file:
                    yaml_data = yaml.safe_load(yaml_file)
                    classes = yaml_data.get('names', [])
            except Exception:
                pass
        with db.session.begin():
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
                    'device': device,
                    'model_type': model_type
                },
                classes=classes
            )
            db.session.add(training_model)
        update_training_info(model_id, {
            'status': 'starting',
            'current_epoch': 0,
            'total_epochs': epochs,
            'progress': 0,
            'metrics': {},
            'device': device,
            'yaml_path': data_yaml_path
        })
        def run_training():
            try:
                print(f"[TRAINING THREAD] Starting YOLO training for model_id={model_id}, model_pt={model_pt}")
                model = YOLO(model_pt)
                print(f"[TRAINING THREAD] YOLO model loaded: {model}")

                # Variável para armazenar o último trainer
                last_trainer = {'obj': None}

                def emit_callback_event(event, trainer, extra=None):
                    last_trainer['obj'] = trainer
                    print("Metrics received:", getattr(trainer, 'metrics', {}))
                    info = {
                        'event': event,
                        'status': 'training',
                        'current_epoch': getattr(trainer, 'epoch', None),
                        'total_epochs': epochs,
                        'metrics': getattr(trainer, 'metrics', {}),
                        'device': device,
                        'yaml_path': data_yaml_path
                    }
                    if extra:
                        info.update(extra)
                    update_training_info(model_id, info)

                def on_pretrain_routine_start(trainer):
                    emit_callback_event('on_pretrain_routine_start', trainer)
                def on_pretrain_routine_end(trainer):
                    emit_callback_event('on_pretrain_routine_end', trainer)
                def on_train_start(trainer):
                    emit_callback_event('on_train_start', trainer)
                # def on_train_epoch_start(trainer):
                #     emit_callback_event('on_train_epoch_start', trainer)
                def on_train_epoch_end(trainer):
                    progress = (trainer.epoch + 1) / epochs
                    emit_callback_event('on_train_epoch_end', trainer, {
                        'progress': progress * 100,
                        'current_epoch': trainer.epoch + 1
                    })
                def on_train_end(trainer):
                    emit_callback_event('on_train_end', trainer, {
                        'status': 'completed',
                        'total_epochs': epochs,
                        'progress': 100,
                        'current_epoch': trainer.epoch + 1
                    })
                    metrics = getattr(trainer, 'metrics', {})
                    print("Metrics received:", metrics)
                    from app import app
                    with app.app_context():
                        with db.session.begin():
                            training_model = db.session.get(YoloModel, model_id)
                            if training_model:
                                training_model.status = 'completed'
                                training_model.progress = 1.0
                                training_model.current_epoch = trainer.epochs
                                training_model.metrics = metrics
                                db.session.add(training_model)

                model.add_callback("on_pretrain_routine_start", on_pretrain_routine_start)
                model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
                model.add_callback("on_train_start", on_train_start)
                # model.add_callback("on_train_epoch_start", on_train_epoch_start)
                model.add_callback("on_train_epoch_end", on_train_epoch_end)
                # model.add_callback("on_model_save", on_model_save)
                model.add_callback("on_train_end", on_train_end)

                results = model.train(
                    data=data_yaml_path,
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=img_size,
                    device=device,
                    project=Config.MODELS_FOLDER,
                    name=model_id,
                    exist_ok=True
                )
                print(f"[TRAINING THREAD] Training finished for model_id={model_id}")
            except Exception as e:
                print(f"[TRAINING THREAD] Error for model_id={model_id}: {str(e)}")
                update_training_info(model_id, {'status': 'error', 'current_epoch': 0, 'total_epochs': epochs, 'progress': 0, 'metrics': {}, 'error_message': str(e)})
        t = threading.Thread(target=run_training, daemon=True)
        t.start()
        return jsonify({'message': 'Training job started', 'model_id': model_id, 'status': 'starting', 'device': device, 'yaml_path': data_yaml_path}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@training_bp.route('/api/training-status', methods=['GET'])
def get_training_status():
    model_id = request.args.get('model_id')
    if model_id:
        info = training_status.get(model_id)
        if info:
            return jsonify({'training_sessions': [{ 'model_id': model_id, 'info': info }]}), 200
        else:
            return jsonify({'error': 'Model not found'}), 404
    else:
        sessions = [{'model_id': k, 'info': v} for k, v in training_status.items()]
        return jsonify({'training_sessions': sessions}), 200

@training_bp.route('/api/stop-training/<model_id>', methods=['POST'])
def stop_training(model_id):
    training_stop_flags[model_id] = True
    update_training_info(model_id, {'status': 'stopped'})
    return jsonify({'message': 'Training stopped', 'model_id': model_id}), 200

@training_bp.route('/api/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    model_id = request.form.get('model_id')
    if not model_id:
        return jsonify({'error': 'No model specified'}), 400
    try:
        model_record = YoloModel.query.get(model_id)
        if not model_record:
            return jsonify({'error': f'Model with ID {model_id} not found in database'}), 404
        model_path = os.path.join(Config.MODELS_FOLDER, f"{model_id}.pt")
        if not os.path.exists(model_path):
            if model_record.file_path and model_record.file_path != f"{model_id}.pt":
                alt_model_path = os.path.join(Config.MODELS_FOLDER, model_record.file_path)
                if os.path.exists(alt_model_path):
                    model_path = alt_model_path
                else:
                    best_path = os.path.join(Config.MODELS_FOLDER, model_id, "weights", "best.pt")
                    if os.path.exists(best_path):
                        model_path = best_path
                    else:
                        return jsonify({'error': 'Model file not found on disk'}), 404
        file = request.files['file']
        confidence = float(request.form.get('confidence', 0.25))
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS:
            filename = secure_filename(file.filename)
            file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(file_path)
            model = YOLO(model_path)
            results = model(file_path, conf=confidence)
            result = results[0]
            detections = []
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [float(coord) for coord in box.xyxy[0].tolist()]
                classes = model_record.classes if model_record.classes else []
                class_name = classes[class_id] if class_id < len(classes) else f"Class {class_id}"
                detections.append({'class_id': class_id, 'class_name': class_name, 'confidence': conf, 'box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'width': x2-x1, 'height': y2-y1}})
            from PIL import Image
            img = Image.open(file_path)
            width, height = img.size
            return jsonify({'detections': detections, 'image_path': f"/api/uploads/{filename}", 'image_dimensions': {'width': width, 'height': height}}), 200
        return jsonify({'error': 'File type not allowed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
