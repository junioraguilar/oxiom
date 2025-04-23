@echo off
echo Starting YOLOv11 Trainer with Conda environment...

REM Create necessary directories if they don't exist
mkdir models 2>nul
mkdir uploads 2>nul
mkdir train_data 2>nul

REM Start backend server with conda
start cmd /k "conda create -y -n yolo-trainer python=3.10 && conda activate yolo-trainer && cd backend && pip install -r requirements.txt && python app.py"

REM Start frontend development server
start cmd /k "cd frontend && npm install && npm run dev"

echo YOLOv11 Trainer servers started!
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo Press any key to shut down all servers...
pause > nul

REM Kill all node and python processes
taskkill /F /IM node.exe >nul 2>&1
taskkill /F /IM python.exe >nul 2>&1 