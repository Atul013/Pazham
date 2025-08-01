# Banana Seed Predictor

A machine learning model that predicts the number of seeds in a banana based on its physical characteristics.

## Overview

This project uses a Random Forest Regressor to predict the number of seeds in a banana based on various physical features. The model achieves approximately 80% accuracy (R² score) on synthetic data and can be retrained with real-world data.

## Features

The model takes into account the following banana characteristics:
- Length (centimeters)
- Width (centimeters)
- Weight (grams)
- Ripeness level (scale 1-5)
- Color (1=green, 2=yellow, 3=brown)
- Curvature (degrees)

## Requirements

- Python 3.x
- Required packages:
  - numpy
  - pandas
  - scikit-learn

## Usage

The model is implemented in a Jupyter notebook (`model.ipynb`). To use it:

1. Open `model.ipynb` in Jupyter or VS Code
2. Run all cells to train the model
3. Use the `predict_seeds()` function with your banana measurements

Example usage:
```python
predicted_seeds = predict_seeds(
    length=16,    # cm
    width=3.2,    # cm
    weight=130,   # g
    ripeness=4,   # scale 1-5
    color=2,      # yellow
    curvature=40  # degrees
)
```

## Model Performance

Current model metrics on synthetic data:
- Mean Squared Error: 0.20
- R² Score: 0.80

Note: These metrics are based on synthetic training data. Performance may vary with real-world data.

## Future Improvements

- Replace synthetic data with real banana measurements
- Add image processing to automatically extract features
- Implement cross-validation
- Add visualization of feature importance
- Create a simple web interface for predictions

## License

[MIT License](LICENSE)
