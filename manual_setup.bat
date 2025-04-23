@echo off
echo Iniciando configuração manual do YOLOv8 Trainer...

REM Criar diretórios necessários
mkdir models 2>nul
mkdir uploads 2>nul
mkdir train_data 2>nul

REM Configuração do backend passo a passo
cd backend

REM Criar e ativar ambiente virtual
echo Criando ambiente virtual...
python -m venv venv
call venv\Scripts\activate

REM Atualizar ferramentas básicas
echo Atualizando pip, setuptools e wheel...
python -m pip install --upgrade pip setuptools wheel

REM Instalar cada dependência separadamente
echo Instalando dependências uma a uma...
pip install flask==2.0.1
pip install flask-cors==3.0.10
pip install pillow==9.0.0
pip install numpy==1.24.3
pip install python-dotenv==0.19.0
pip install werkzeug==2.0.2

echo Deseja instalar PyTorch com suporte a CUDA? (S/N)
choice /C SN /M "Instalar com CUDA"
if errorlevel 2 (
    REM Instalar PyTorch sem CUDA
    echo Instalando PyTorch CPU...
    pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu
) else (
    REM Instalar PyTorch com CUDA
    echo Instalando PyTorch com suporte a CUDA...
    pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu117
)

REM Instalar ultralytics
echo Instalando Ultralytics...
pip install ultralytics==8.0.147

echo Instalação do backend concluída!
echo Para iniciar o backend, execute: python app.py

REM Volta para o diretório raiz
cd ..

REM Configuração do frontend
cd frontend
echo Instalando dependências do frontend...
call npm install

echo Configuração concluída! Para iniciar:
echo - Backend: cd backend ^&^& venv\Scripts\activate ^&^& python app.py
echo - Frontend: cd frontend ^&^& npm run dev

pause 