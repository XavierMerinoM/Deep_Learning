# Time Series-Based Prediction of Fruit and Vegetable Market Prices

## Summary of Project

This project implements and compares two deep learning models to predict the average market prices of fruits and vegetables using time series data. The models are:

1. **Baseline CNN Model**: A convolutional neural network for time series regression.
2. **Improved LSTM-CNN Hybrid Model**: Combines bidirectional LSTM and CNN layers with attention mechanisms for enhanced prediction.

The pipeline includes preprocessing, training, evaluation, hyperparameter optimization, and explainability using SHAP values.

---

## Preprocessing Steps

- **Dataset**: [Kalimati Tarkari Dataset from Kaggle](https://www.kaggle.com/datasets/ramkrijal/agriculture-vegetables-fruits-time-series-prices)
- **Data cleaning**:
  - Removed irrelevant columns (e.g., SN, Unit).
  - Filtered top 10 commodities with sufficient data.
  - Converted date column to datetime format.
  - Extracted temporal features (year, month, day, day_of_week).
- **Feature engineering**:
  - Price change tracking.
  - Seasonal features using sine/cosine transformations.
  - Price range and position features.
- **Normalization**: RobustScaler and StandardScaler used per commodity.
- **Sequence generation**: 30-day sequences created for each commodity.
- **Data splitting**: Commodity-aware split into train (60%), validation (20%), and test (20%).

---

## Model Architecture

### Model 1: Baseline CNN

- 2 Conv1D blocks with batch normalization and dropout.
- Global average pooling.
- Dense layers for regression.
- Output: single price prediction.

### Model 2: LSTM-CNN Hybrid

- Bidirectional LSTM layer.
- 2 Conv1D layers with batch normalization and dropout.
- Attention mechanism.
- Global average pooling and dense layers.
- Output: single price prediction.

---

## Training Process

### Framework

- TensorFlow and Keras.
- SHAP for explainability.
- Training on Google Colab with GPU.

### Loss Function

- Huber loss (robust to outliers).

### Optimizer

- Adam optimizer with AMSGrad.
- Learning rate scheduling and early stopping.

### Metrics

- Mean Absolute Error (MAE).
- Validation MAE tracked across epochs.

---

## Results

### Baseline CNN Model

- **MAE**: 20.49
- **SMAPE**: 32.68%
- **MASE**: 6.9777

### Improved LSTM-CNN Model

- **MAE**: 8.39
- **SMAPE**: 11.62%
- **MASE**: 2.8564

### Optimized CNN Model

- **MAE**: 15.08
- **SMAPE**: 21.52%
- **MASE**: 5.1365

### Optimized LSTM-CNN Model

- **MAE**: 8.80
- **SMAPE**: 12.07%
- **MASE**: 2.9983

---

## Explainability

- SHAP values computed using GradientExplainer.
- Feature importance plots and force plots generated.
- Top contributing features identified for individual predictions.
- Features include temporal signals, price changes, and seasonal encodings.

