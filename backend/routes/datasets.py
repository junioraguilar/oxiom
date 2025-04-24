from flask import Blueprint, request, jsonify
import os
import uuid
from werkzeug.utils import secure_filename
from extensions import db
from config import Config
from models import Dataset
from flask import send_file

datasets_bp = Blueprint('datasets', __name__)

# Dataset-related endpoints will be moved here from app.py

@datasets_bp.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.zip'):
        try:
            dataset_id = str(uuid.uuid4())
            dataset_dir = os.path.join(Config.TRAIN_DATA_FOLDER, dataset_id)
            os.makedirs(dataset_dir, exist_ok=True)
            filename = secure_filename(file.filename)
            file_path = os.path.join(dataset_dir, filename)
            file.save(file_path)
            import zipfile
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            # Find YAML
            yaml_path = None
            for root, dirs, files in os.walk(dataset_dir):
                for f in files:
                    if f.lower() == 'data.yaml':
                        yaml_path = os.path.join(root, f)
                        break
                if yaml_path:
                    break
            if not yaml_path:
                for root, dirs, files in os.walk(dataset_dir):
                    for f in files:
                        if f.lower().endswith('.yaml'):
                            yaml_path = os.path.join(root, f)
                            import shutil
                            shutil.copy(yaml_path, os.path.join(dataset_dir, 'data.yaml'))
                            yaml_path = os.path.join(dataset_dir, 'data.yaml')
                            break
                    if yaml_path:
                        break
            data_yaml_path = os.path.join(dataset_dir, 'data.yaml')
            if not os.path.exists(data_yaml_path) and yaml_path:
                import shutil
                shutil.copy(yaml_path, data_yaml_path)
            if not os.path.exists(data_yaml_path):
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
            classes = []
            try:
                import yaml
                with open(data_yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    if 'names' in yaml_data:
                        classes = yaml_data['names']
            except Exception as e:
                print(f"Error extracting classes from YAML: {str(e)}")
            dataset_name = os.path.splitext(filename)[0]
            file_size = os.path.getsize(file_path)
            with db.session.begin():
                dataset = Dataset(
                    id=dataset_id,
                    name=dataset_name,
                    file_path=os.path.relpath(file_path, Config.TRAIN_DATA_FOLDER),
                    yaml_path=os.path.relpath(data_yaml_path, Config.TRAIN_DATA_FOLDER),
                    file_size=file_size,
                )
                dataset.classes = classes
                db.session.add(dataset)
            return jsonify({'message': 'Dataset uploaded successfully', 'dataset_id': dataset_id, 'name': dataset_name, 'path': dataset_dir, 'yaml_path': data_yaml_path, 'classes': classes}), 200
        except Exception as e:
            print(f"Error during dataset upload: {str(e)}")
            return jsonify({'error': f'Error processing dataset: {str(e)}'}), 500
    return jsonify({'error': 'File type not allowed. Please upload a ZIP file'}), 400

@datasets_bp.route('/api/datasets', methods=['GET'])
def list_datasets():
    try:
        db_datasets = Dataset.query.order_by(Dataset.created_at.desc()).all()
        datasets = [dataset.to_dict() for dataset in db_datasets]
        return jsonify({'datasets': datasets}), 200
    except Exception as e:
        print(f"Error listing datasets: {str(e)}")
        return jsonify({'error': str(e)}), 500

@datasets_bp.route('/api/delete-dataset/<dataset_id>', methods=['DELETE'])
def delete_dataset(dataset_id):
    try:
        dataset = Dataset.query.get(dataset_id)
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        # Remove files
        dataset_dir = os.path.join(Config.TRAIN_DATA_FOLDER, dataset.id)
        if os.path.exists(dataset_dir):
            import shutil
            shutil.rmtree(dataset_dir)
        db.session.delete(dataset)
        db.session.commit()
        return jsonify({'message': 'Dataset deleted successfully'}), 200
    except Exception as e:
        print(f"Error deleting dataset: {str(e)}")
        return jsonify({'error': str(e)}), 500

@datasets_bp.route('/api/download-dataset-file/<dataset_id>', methods=['GET'])
def download_dataset_file(dataset_id):
    try:
        dataset = Dataset.query.get(dataset_id)
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        dataset_dir = os.path.join(Config.TRAIN_DATA_FOLDER, dataset.id)
        file_path = os.path.join(dataset_dir, os.path.basename(dataset.file_path))
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        print(f"Error downloading dataset file: {str(e)}")
        return jsonify({'error': str(e)}), 500
