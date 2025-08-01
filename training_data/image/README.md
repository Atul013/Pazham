# Images Directory

This directory contains the banana images used for training the Banana Feature Prediction model.

## Image Requirements

1. **Format**: JPG or PNG files
2. **Resolution**: 224x224 pixels (standard for vision models)
3. **Perspective**: Side view of the banana
4. **Background**: Preferably solid color or simple background
5. **Lighting**: Consistent, well-lit conditions
6. **File naming**: Using the pattern `p[number].(jpg|jpeg)`

## Image Usage

These images are used in conjunction with the dataset CSV file, which contains corresponding measurements and features for each image. The trained model learns to predict:

- Number of seeds
- Curvature angle

## Data Quality

- Images should be clear and in focus
- Banana should be clearly visible against the background
- Consistent lighting conditions help improve model accuracy
- Side view perspective ensures consistent curvature measurements
