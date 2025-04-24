import os
from flask import Flask
from config import Config
from extensions import db, cors, socketio
from models import YoloModel, Dataset
from routes.training import training_bp
from routes.models import models_bp
from routes.datasets import datasets_bp
from routes.uploads import uploads_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    cors.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")
    app.register_blueprint(training_bp)
    app.register_blueprint(models_bp)
    app.register_blueprint(datasets_bp)
    app.register_blueprint(uploads_bp)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)
    os.makedirs(app.config['TRAIN_DATA_FOLDER'], exist_ok=True)
    with app.app_context():
        db.create_all()
        models_in_progress = YoloModel.query.filter(YoloModel.status.in_(['training', 'starting', 'running'])).all()
        for model in models_in_progress:
            model.status = 'error'
            model.error_message = 'Training interrupted by backend restart.'
            db.session.commit()
    return app

app = create_app()

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)