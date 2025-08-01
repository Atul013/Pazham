# Training Data Directory

This directory contains the training data and images for the Banana Feature Prediction model.

## Directory Structure

- `dataset.csv`: Original dataset containing banana measurements and features
- `clean_dataset.csv`: Cleaned version of the dataset with validated image paths
- `image/`: Directory containing banana images used for training
- `a.txt`: Additional data file (if used)

## Dataset Format

The dataset CSV files contain the following columns:

1. `image_filename`: Name of the corresponding image file
2. `length_cm`: Length of the banana in centimeters
3. `width_cm`: Width at the middle point in centimeters
4. `weight_g`: Weight in grams
5. `ripeness`: Scale from 1-5 (1: Very unripe to 5: Overripe)
6. `color_code`: Color encoded as 1=green, 2=yellow, 3=brown
7. `seed_count`: Actual number of seeds
8. `curvature_degrees`: Measured angle of curvature

## Image Requirements

- Format: JPG or PNG
- Resolution: 224x224 pixels
- Perspective: Side view of the banana
- Background: Preferably solid color or simple background
- Lighting: Consistent, well-lit conditions
