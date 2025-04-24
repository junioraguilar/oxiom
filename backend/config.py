import os

class Config:
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
    MODELS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models_saved')
    TRAIN_DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'train_data')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'zip', 'pt'}
    DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_db.sqlite')
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{DATABASE_PATH}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1GB
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
