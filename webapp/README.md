# Web Application Directory

This directory contains the Flask web application for the Banana Feature Prediction model.

## Directory Structure

- `app.py`: Main Flask application file
- `model_utils.py`: Utility functions for model loading and predictions
- `static/`: Static files (CSS, JavaScript, images)
- `templates/`: HTML templates
- `uploads/`: Directory for temporarily storing uploaded images
- `__pycache__/`: Python cache directory (automatically generated)

## Components

### Backend (`app.py`)

- Flask application setup and routing
- File upload handling
- Model prediction integration

### Model Utils (`model_utils.py`)

- Model loading and initialization
- Image preprocessing
- Prediction functions

### Templates

- `index.html`: Main page with upload form
- `result.html`: Displays prediction results

### Static Files

- CSS styles
- JavaScript for client-side functionality
- Image assets

## Usage

1. Ensure the trained model file (`best_model.pth`) is in the parent directory
2. Run the Flask application:
   ```bash
   python app.py
   ```
3. Open a web browser and navigate to `http://localhost:5000`
4. Upload a banana image and get predictions for:
   - Number of seeds
   - Curvature in degrees
