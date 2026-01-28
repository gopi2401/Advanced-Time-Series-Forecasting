# Advanced Time Series Forecasting using LSTM with Self-Attention

## Overview

This project implements an advanced deep learning approach for time series forecasting using an LSTM network enhanced with a **custom self-attention mechanism**.
The goal is to improve forecasting performance and interpretability by allowing the model to learn which past time steps are most important for prediction.

The project strictly follows time series evaluation best practices, including **rolling origin cross-validation, and provides direct visualization of learned attention weights**.

## Key Features

- LSTM-based sequence modeling
- Custom self-attention layer implemented from scratch
- Rolling origin cross-validation for time series evaluation
- Visualization of temporal attention weights
- Comparison between actual and predicted values
- Fully reproducible synthetic dataset

## Model Architecture

The model consists of three main components:

1. LSTM Layer
   Captures temporal dependencies in the input sequence.

2. Self-Attention Layer
   Learns a weight for each time step in the LSTM output sequence and produces a context vector as a weighted sum.

3. Fully Connected Output Layer
   Maps the attention-based context vector to the final prediction.

Unlike standard LSTM models, the final prediction is **not** taken from the last hidden state. Instead, it is derived from the attention-weighted context vector.

## Attention Mechanism Explanation

The self-attention layer computes a score for each time step in the LSTM output.
These scores are normalized using a softmax function to produce attention weights.

Higher attention weights indicate that the model considers those time steps more important for forecasting.
The learned weights are visualized to interpret the temporal focus of the model.

## Dataset

A synthetic multivariate time series dataset is generated with:

- Two input features based on sine and cosine functions with noise
- One target variable constructed from the features and a time trend

The dataset is scaled using MinMax normalization before training.

## Evaluation Strategy

### Rolling Origin Cross-Validation

Instead of a standard train-test split, the model is evaluated using **rolling origin cross-validation**, which is appropriate for time series data.

For each fold:

- Training data includes all observations up to a given time point
- Testing data includes the immediately following time segment
- RMSE is computed for each fold

This approach prevents data leakage and simulates real-world forecasting scenarios.

## Results

- Root Mean Squared Error (RMSE) is reported for each rolling fold
- Average RMSE is used as the final performance metric
- Attention weight plots demonstrate that the model focuses more on recent time steps

## Visualizations

The project generates the following plots:

- Learned temporal attention weights for a sample prediction
- Actual vs predicted target values over time

These visualizations help validate both performance and interpretability.

## Requirements

- Python 3.8 or higher
- NumPy
- Pandas
- PyTorch
- scikit-learn
- Matplotlib

Install dependencies using:

`pip install numpy pandas torch scikit-learn matplotlib`

## How to Run

1. Clone the repository or download the script.
2. Ensure all required libraries are installed.
3. Run the Python script:

`python main.py`

All results and visualizations will be displayed automatically.

## Project Structure

```.
├── main.py
├── README.md
```

## Notes

- No external explainability tools such as SHAP are used.
- Attention weights are directly extracted from the model.
- The implementation prioritizes clarity, correctness, and academic integrity.

## Author

Developed as part of an advanced time series forecasting exercise using deep learning and attention mechanisms.
