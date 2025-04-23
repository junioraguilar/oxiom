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
# Instala torch GPU separadamente
RUN /opt/conda/envs/yolov8-backend/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
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

EXPOSE 5000
CMD ["python", "backend/app.py"]
