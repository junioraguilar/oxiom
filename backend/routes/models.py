from flask import Blueprint, request, jsonify, send_file
import os
import uuid
from werkzeug.utils import secure_filename
from extensions import db
from config import Config
from models import YoloModel

models_bp = Blueprint('models', __name__)

# Model-related endpoints will be moved here from app.py

@models_bp.route('/api/models', methods=['GET'])
def list_models():
    """List all available models from the database, checking all possible locations (upload, direct, training)."""
    try:
        db_models = YoloModel.query.filter(
            YoloModel.status.in_(['completed', 'stopped'])
        ).order_by(YoloModel.created_at.desc()).all()
        models = []
        for model in db_models:
            model_path = None
            model_exists = False
            # 1. Check file_path (upload pattern: <uuid>_<name>.pt)
            if model.file_path:
                full_path = os.path.join(Config.MODELS_FOLDER, model.file_path)
                if os.path.exists(full_path):
                    model_path = full_path
                    model_exists = True
            # 2. Check <id>.pt (direct save)
            if not model_exists:
                direct_path = os.path.join(Config.MODELS_FOLDER, f"{model.id}.pt")
                if os.path.exists(direct_path):
                    model_path = direct_path
                    model_exists = True
            # 3. Check <id>/weights/best.pt (ultralytics train)
            if not model_exists:
                best_path = os.path.join(Config.MODELS_FOLDER, model.id, "weights", "best.pt")
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
                    'classes': model.classes,
                    'metrics': model.metrics if hasattr(model, 'metrics') else {},
                    'status': model.status,
                    'progress': model.progress * 100 if model.progress is not None else 100,
                    'epochs': model.total_epochs or 0,
                    'completed_epochs': model.current_epoch or model.total_epochs or 0
                })
        return jsonify({'models': models}), 200
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@models_bp.route('/api/upload-model', methods=['POST'])
def upload_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    model_name = request.form.get('name', 'Uploaded model')
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS:
        model_id = str(uuid.uuid4())
        filename = f"{model_id}_" + secure_filename(file.filename)
        file_path = os.path.join(Config.MODELS_FOLDER, filename)
        file.save(file_path)
        file_size = os.path.getsize(file_path)
        with db.session.begin():
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
        return jsonify({'message': 'Model uploaded successfully', 'model_id': model_id, 'filename': filename}), 201
    return jsonify({'error': 'File type not allowed'}), 400

@models_bp.route('/api/trained-models', methods=['GET'])
def list_trained_models():
    """List trained models with their details, checking all possible locations (upload, direct, training)."""
    try:
        db_models = YoloModel.query.order_by(YoloModel.created_at.desc()).all()
        trained_models = []
        for model in db_models:
            model_path = os.path.join(Config.MODELS_FOLDER, f"{model.id}.pt")
            best_model_path = os.path.join(Config.MODELS_FOLDER, model.id, "weights", "best.pt")
            file_path = None
            model_file_exists = os.path.exists(model_path)
            best_model_exists = os.path.exists(best_model_path)
            # Prefer uploaded or direct .pt, fallback to best.pt
            if model_file_exists:
                file_path = model_path
            elif best_model_exists:
                file_path = best_model_path
            elif model.file_path:
                alt_path = os.path.join(Config.MODELS_FOLDER, model.file_path)
                if os.path.exists(alt_path):
                    file_path = alt_path
            if file_path and os.path.exists(file_path):
                trained_models.append({
                    'id': model.id,
                    'name': model.name or f"Model {model.id[:8]}",
                    'path': file_path,
                    'size': os.path.getsize(file_path) if file_path and os.path.exists(file_path) else model.file_size,
                    'created_at': model.created_at.isoformat() if model.created_at else None,
                    'classes': model.classes,
                    'status': model.status or 'COMPLETED',
                    'progress': (model.progress * 100) if model.progress is not None else 100,
                    'epochs': model.total_epochs or 0,
                    'completed_epochs': model.current_epoch or model.total_epochs or 0,
                    'metrics': model.metrics or {}
                })
        return jsonify({'trained_models': trained_models}), 200
    except Exception as e:
        print(f"Error listing trained models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@models_bp.route('/api/download-model/<model_id>', methods=['GET'])
def download_model(model_id):
    try:
        model = YoloModel.query.get(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        model_path = os.path.join(Config.MODELS_FOLDER, f"{model_id}.pt")
        best_model_path = os.path.join(Config.MODELS_FOLDER, model_id, "weights", "best.pt")
        if os.path.exists(model_path):
            path = model_path
        elif os.path.exists(best_model_path):
            path = best_model_path
        else:
            return jsonify({'error': 'Model file not found'}), 404
        return send_file(path, as_attachment=True)
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@models_bp.route('/api/delete-model/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    try:
        # Use a new session for deletion to avoid session conflicts
        from extensions import db as db_ext
        model = db_ext.session.get(YoloModel, model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        # Remove files
        model_path = os.path.join(Config.MODELS_FOLDER, f"{model_id}.pt")
        best_model_path = os.path.join(Config.MODELS_FOLDER, model_id, "weights", "best.pt")
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(best_model_path):
            import shutil
            shutil.rmtree(os.path.dirname(os.path.dirname(best_model_path)), ignore_errors=True)
        db_ext.session.delete(model)
        db_ext.session.commit()
        return jsonify({'message': 'Model deleted successfully'}), 200
    except Exception as e:
        print(f"Error deleting model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@models_bp.route('/api/debug/list-model-files', methods=['GET'])
def debug_list_model_files():
    import glob
    import os
    files = []
    # List all .pt files in the models folder (flat only)
    pattern = os.path.join(Config.MODELS_FOLDER, '*.pt')
    for filepath in glob.glob(pattern):
        files.append({
            'filename': os.path.basename(filepath),
            'full_path': filepath,
            'size': os.path.getsize(filepath)
        })
    # Optionally, list all files recursively (including subfolders)
    for dirpath, dirnames, filenames in os.walk(Config.MODELS_FOLDER):
        for fname in filenames:
            if fname.endswith('.pt') and os.path.join(Config.MODELS_FOLDER, fname) not in [f['full_path'] for f in files]:
                full_path = os.path.join(dirpath, fname)
                files.append({
                    'filename': fname,
                    'full_path': full_path,
                    'size': os.path.getsize(full_path)
                })
    return jsonify({'model_files': files}), 200

@models_bp.route('/api/debug/compare-models', methods=['GET'])
def debug_compare_models():
    import glob
    import os
    from models import YoloModel
    # Get all .pt files in the models folder (recursively)
    real_files = set()
    for dirpath, dirnames, filenames in os.walk(Config.MODELS_FOLDER):
        for fname in filenames:
            if fname.endswith('.pt'):
                real_files.add(os.path.abspath(os.path.join(dirpath, fname)))
    # Get all file_paths from DB
    db_models = YoloModel.query.all()
    db_files = set()
    db_info = []
    for model in db_models:
        if model.file_path:
            if os.path.isabs(model.file_path):
                full_path = model.file_path
            else:
                full_path = os.path.abspath(os.path.join(Config.MODELS_FOLDER, model.file_path))
            db_files.add(full_path)
            db_info.append({'id': model.id, 'file_path': model.file_path, 'full_path': full_path})
    # Files in DB but not on disk
    missing_on_disk = [info for info in db_info if info['full_path'] not in real_files]
    # Files on disk but not in DB
    extra_on_disk = [f for f in real_files if f not in db_files]
    return jsonify({
        'missing_on_disk': missing_on_disk,
        'extra_on_disk': extra_on_disk,
        'db_files': list(db_files),
        'real_files': list(real_files)
    }), 200
