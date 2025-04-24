from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import uuid
from extensions import db
from config import Config
from models import YoloModel

uploads_bp = Blueprint('uploads', __name__)

# Helper to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@uploads_bp.route('/api/upload-image', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        return jsonify({'message': 'File uploaded successfully', 'filename': unique_filename, 'path': file_path}), 200
    return jsonify({'error': 'File type not allowed'}), 400

@uploads_bp.route('/api/uploads/<filename>', methods=['GET'])
def get_uploaded_file(filename):
    return send_from_directory(Config.UPLOAD_FOLDER, filename)
