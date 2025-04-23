# YOLOv8 Trainer Web Interface

This project provides a web interface for training, uploading, and testing YOLOv8 models.

## Features

- Upload and train custom YOLOv8 models
- Upload pre-trained YOLOv8 models
- Test object detection by uploading images
- View detection results with bounding boxes and confidence scores

## Project Structure

```
├── backend/            # Flask backend API
│   ├── app.py          # Main Flask application  
│   └── requirements.txt # Python dependencies
├── frontend/           # React frontend
│   ├── public/         # Static assets
│   └── src/            # React source code
├── models/             # Stored models
├── uploads/            # Uploaded images
└── train_data/         # Training datasets
```

## Setup Instructions

### Prerequisites

- Python 3.8+ 
- Node.js 16+ 
- npm or yarn

### Backend Setup (Flask)

1. Open a terminal and navigate to the backend directory:

```
cd backend
```

2. Create a virtual environment (recommended):

```
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

4. Start the Flask server:

```
python app.py
```

The backend server will start on http://localhost:5000

### Frontend Setup (React)

1. Open a new terminal and navigate to the frontend directory:

```
cd frontend
```

2. Install dependencies:

```
npm install
```

3. Start the development server:

```
npm run dev
```

The frontend development server will start on http://localhost:3000

## Usage

1. Open your browser and go to http://localhost:3000
2. Use the navigation menu to access different features:
   - **Train Model**: Upload a dataset and train a custom YOLOv8 model
   - **Test Model**: Select a model and upload images for object detection
   - **Upload Model**: Upload pre-trained YOLOv8 model files (.pt)

## Training Data Format

The training data should be in YOLO format, typically a ZIP file containing:

- `images/` directory with training images
- `labels/` directory with corresponding label files
- `data.yaml` configuration file defining classes

## Notes

- The actual model training is resource-intensive and may require a GPU.
- For production use, consider implementing background job processing for training tasks.
- Large model files may require adjusting the server's file size limits. 