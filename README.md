<p align="center">
  <img src="frontend/logo.png" alt="Oxiom Logo" width="360" />
</p>

<p align="center"><b>Oxiom - Train, test, and upload models for object detection with an easy-to-use interface</b></p>

---

This project provides a web interface for training, uploading, and testing YOLOv8 models.

## Features

- Upload and train custom YOLOv8 models
- Upload pre-trained YOLOv8 models
- Object detection testing via image upload
- Visualization of detection results with bounding boxes and confidence scores

## Project Structure

```
├── backend/            # Flask API backend
│   ├── app.py          # Main Flask application  
│   └── requirements.txt # Python dependencies
├── frontend/           # React frontend
│   ├── public/         # Static assets
│   └── src/            # React source code
├── models/             # Stored models (gitignored)
├── uploads/            # Uploaded images (gitignored)
└── train_data/         # Training datasets (gitignored)
```

## Version Control

This project uses Git for version control, with the following settings:

- Files ignored by Git (set in `.gitignore`):
  - `models/` directory (large model files)
  - `train_data/` directory (training datasets)
  - `uploads/` directory (user-uploaded images)
  - `.pt` files (PyTorch model files)
  - `node_modules/` directory (Node.js dependencies)
  - Temporary and cache files

## How to Run (Development)

### Prerequisites

- [Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Node.js 18+ and npm
- Docker and Docker Compose (optional for containerized setup)
- NVIDIA GPU + drivers + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU support

### Backend (Flask + YOLOv8) with Conda

1. Create the Conda environment from the `environment.yml` file in the project root:

```
conda env create -f environment.yml
conda activate yolov8-backend
```

2. Install PyTorch with CUDA (GPU) support manually (recommended):

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

3. Install the remaining dependencies:

```
pip install -r backend/requirements.txt
```

4. Start the Flask API:

```
cd backend
python app.py
```

### Frontend (React)

1. Install dependencies:

```
cd frontend
npm install
```

2. Start the frontend in development mode:

```
npm run dev
```

The frontend will be available at http://localhost:5173

### Docker Environment (optional)

You can run the entire project using Docker Compose (with GPU support):

```
docker compose up --build
```

This will start the backend (Flask + YOLOv8 + Conda) and frontend (React) in isolated containers. The backend will use GPU if available.

## Notes
- You do not need to use `.bat` scripts to activate environments.
- For production, adjust environment variables and settings as needed.
- Make sure your GPU and drivers are properly set up for CUDA usage.

## Contact

Questions or suggestions? Open an issue in the repository!