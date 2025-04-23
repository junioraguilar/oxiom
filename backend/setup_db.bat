@echo off
echo Instalando dependencias adicionais...
pip install SQLAlchemy==1.4.46 Flask-SQLAlchemy==2.5.1

echo Inicializando o banco de dados...
python -c "from app import app, db; app.app_context().push(); db.create_all()"

echo Migrando modelos existentes...
python migrate_models.py

echo Configuracao concluida com sucesso!
echo O banco de dados SQLite foi criado e os modelos existentes foram migrados.
pause 