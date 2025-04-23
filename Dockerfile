# Backend + Frontend monorepo Dockerfile
# Backend (Flask) with Conda
FROM continuumio/miniconda3:4.12.0 as backend
WORKDIR /app
COPY environment.yml ./
COPY backend/requirements.txt ./requirements.txt
RUN conda env create -f environment.yml && conda clean -afy
SHELL ["/bin/bash", "-c"]
# Activate env and install backend
RUN echo "conda activate yolov8-backend" > ~/.bashrc
ENV PATH /opt/conda/envs/yolov8-backend/bin:$PATH
# Install torch GPU separately
RUN /opt/conda/envs/yolov8-backend/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# Install OpenCV dependencies for libGL
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
WORKDIR /app/backend
COPY backend .

# Frontend (Node)
FROM node:20-slim as frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --legacy-peer-deps --no-audit --progress=false
COPY frontend .
RUN npm run build || echo 'Frontend build failed, continue for backend-only'

# Final image
FROM continuumio/miniconda3:4.12.0
WORKDIR /app
COPY --from=backend /opt/conda /opt/conda
COPY --from=backend /app/backend ./backend
COPY --from=frontend /app/frontend ./frontend
ENV PATH /opt/conda/envs/yolov8-backend/bin:$PATH
# Install OpenCV dependencies for libGL in final image
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

EXPOSE 5000
CMD ["python", "backend/app.py"]
