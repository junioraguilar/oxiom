"""
Script para migrar modelos existentes para o banco de dados SQLite.
Execute este script após atualizar o código para usar SQLite.
"""

import os
import sys
import time
import json
import uuid
from datetime import datetime
from pathlib import Path

# Adicionar diretório pai ao path para importação correta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from models import db, YoloModel

# Configuração básica da aplicação Flask
app = Flask(__name__)
MODELS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_db.sqlite')

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DATABASE_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inicializar o banco de dados
db.init_app(app)

def scan_models_directory():
    """Escaneia o diretório de modelos e retorna uma lista de arquivos de modelo"""
    model_files = []
    
    # Listar arquivos .pt diretamente no diretório models
    for item in os.listdir(MODELS_FOLDER):
        if item.endswith('.pt'):
            file_path = os.path.join(MODELS_FOLDER, item)
            if os.path.isfile(file_path):
                model_files.append({
                    'path': file_path,
                    'filename': item
                })
    
    # Procurar por modelos em estrutura de diretórios (do treinamento)
    for item in os.listdir(MODELS_FOLDER):
        model_dir = os.path.join(MODELS_FOLDER, item)
        if os.path.isdir(model_dir):
            weights_dir = os.path.join(model_dir, 'weights')
            if os.path.isdir(weights_dir):
                best_model = os.path.join(weights_dir, 'best.pt')
                if os.path.isfile(best_model):
                    model_files.append({
                        'path': best_model,
                        'filename': f"{item}/weights/best.pt",
                        'id': item  # O diretório é o ID neste caso
                    })
    
    return model_files

def extract_model_id(filename):
    """Extrai o ID do modelo do nome do arquivo"""
    # Padrão: UUID_nomedoarquivo.pt ou UUID.pt
    parts = filename.split('_', 1)
    if len(parts) > 0:
        potential_id = parts[0]
        # Verificar se é UUID válido
        try:
            uuid_obj = uuid.UUID(potential_id)
            return str(uuid_obj)
        except ValueError:
            pass
    
    # Se não conseguir extrair, gerar um novo UUID
    return str(uuid.uuid4())

def migrate_models():
    """Migra os modelos existentes para o banco de dados SQLite"""
    with app.app_context():
        # Criar tabelas se não existirem
        db.create_all()
        
        # Listar modelos já cadastrados para evitar duplicação
        existing_models = {model.id: model for model in YoloModel.query.all()}
        existing_paths = {model.file_path for model in existing_models.values()}
        
        # Escanear diretório de modelos
        model_files = scan_models_directory()
        print(f"Encontrados {len(model_files)} arquivo(s) de modelo")
        
        # Migrar cada modelo
        for model_file in model_files:
            file_path = model_file['path']
            filename = model_file['filename']
            
            # Determinar o ID do modelo
            if 'id' in model_file:
                model_id = model_file['id']
            else:
                model_id = extract_model_id(filename)
            
            # Verificar se o modelo já existe no banco
            if model_id in existing_models:
                print(f"Modelo {model_id} já existe no banco, atualizando...")
                model = existing_models[model_id]
            elif filename in existing_paths:
                print(f"Arquivo {filename} já está associado a outro modelo, pulando...")
                continue
            else:
                print(f"Adicionando novo modelo {model_id} ({filename})...")
                model = YoloModel(
                    id=model_id,
                    file_path=filename
                )
                db.session.add(model)
            
            # Atualizar os dados do modelo
            model.status = 'completed'  # Assumimos que modelos existentes estão completos
            model.progress = 1.0
            model.file_size = os.path.getsize(file_path)
            
            # Se não tiver nome, usar parte do nome do arquivo
            if not model.name:
                name_parts = filename.split('_', 1)
                if len(name_parts) > 1:
                    model.name = name_parts[1].replace('.pt', '')
                else:
                    model.name = f"Model {model_id[:8]}"
            
            # Se não tiver data de criação, usar a data de modificação do arquivo
            if not model.created_at:
                mtime = os.path.getmtime(file_path)
                model.created_at = datetime.fromtimestamp(mtime)
        
        # Salvar mudanças
        db.session.commit()
        print("Migração concluída com sucesso!")

if __name__ == "__main__":
    print("Iniciando migração de modelos para o banco de dados...")
    migrate_models()
    print("Migração concluída!") 