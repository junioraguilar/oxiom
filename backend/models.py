from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class YoloModel(db.Model):
    """Modelo de dados para armazenar informações sobre modelos YOLOv8 treinados"""
    
    __tablename__ = 'yolo_models'
    
    id = db.Column(db.String(36), primary_key=True)  # UUID
    name = db.Column(db.String(100), nullable=True)
    file_path = db.Column(db.String(255), nullable=False)  # Caminho relativo no sistema de arquivos
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    dataset_id = db.Column(db.String(36), nullable=True)  # Referência ao dataset usado
    status = db.Column(db.String(20), default='training')  # training, completed, error, stopped
    
    # Armazenar como JSON serializados
    _metrics = db.Column(db.Text, nullable=True)
    _parameters = db.Column(db.Text, nullable=True)
    _classes = db.Column(db.Text, nullable=True)
    
    # Informações de progresso
    current_epoch = db.Column(db.Integer, default=0)
    total_epochs = db.Column(db.Integer, default=0)
    progress = db.Column(db.Float, default=0.0)
    file_size = db.Column(db.Integer, default=0)  # Tamanho em bytes
    error_message = db.Column(db.Text, nullable=True)
    
    @property
    def metrics(self):
        if self._metrics:
            return json.loads(self._metrics)
        return {}
    
    @metrics.setter
    def metrics(self, value):
        self._metrics = json.dumps(value) if value else None
    
    @property
    def parameters(self):
        if self._parameters:
            return json.loads(self._parameters)
        return {}
    
    @parameters.setter
    def parameters(self, value):
        self._parameters = json.dumps(value) if value else None
    
    @property
    def classes(self):
        if self._classes:
            return json.loads(self._classes)
        return []
    
    @classes.setter
    def classes(self, value):
        self._classes = json.dumps(value) if value else None
    
    def to_dict(self):
        """Converte o modelo para um dicionário para serialização em JSON"""
        return {
            'id': self.id,
            'name': self.name,
            'status': self.status,
            'progress': self.progress * 100,  # Converter para porcentagem
            'epochs': self.total_epochs,
            'completed_epochs': self.current_epoch,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'dataset_id': self.dataset_id,
            'file_size': self.file_size,
            'metrics': self.metrics,
            'parameters': self.parameters,
            'classes': self.classes,
            'error_message': self.error_message
        } 