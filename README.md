# Bank Marketing Classification with Deep Neural Networks

A machine learning project to predict whether bank clients will subscribe to a term deposit using various classification models including Deep Neural Networks (DNN), XGBoost, and LightGBM.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [File Structure](#file-structure)
- [Contributing](#contributing)

## Project Overview

This project implements multiple machine learning models to solve a binary classification problem in banking: predicting whether a client will subscribe to a term deposit based on their demographic and campaign interaction data.

**Key Models Implemented:**
- Deep Neural Network (DNN) with TensorFlow/Keras
- XGBoost Classifier (with hyperparameter tuning)
- LightGBM Classifier (with hyperparameter tuning)

## Dataset Description

The dataset contains information about bank marketing campaigns with the following characteristics:

**Target Variable:**
- `y`: Whether the client subscribed to a term deposit (binary: 1=yes, 0=no)

**Features (16 total):**

### Demographic Information
- `age`: Age of the client (numeric)
- `job`: Type of job (categorical)
- `marital`: Marital status (categorical: married, single, divorced)
- `education`: Level of education (categorical: primary, secondary, tertiary, unknown)

### Financial Information
- `default`: Has credit in default? (categorical: yes, no)
- `balance`: Average yearly balance in euros (numeric)
- `housing`: Has a housing loan? (categorical: yes, no)
- `loan`: Has a personal loan? (categorical: yes, no)

### Campaign Information
- `contact`: Type of communication contact (categorical: unknown, telephone, cellular)
- `day`: Last contact day of the month (numeric, 1-31)
- `month`: Last contact month of the year (categorical: jan, feb, mar, ..., dec)
- `duration`: Last contact duration in seconds (numeric)
- `campaign`: Number of contacts performed during this campaign (numeric)
- `pdays`: Number of days since client was last contacted from previous campaign (numeric; -1 = not previously contacted)
- `previous`: Number of contacts performed before this campaign (numeric)
- `poutcome`: Outcome of previous marketing campaign (categorical: unknown, other, failure, success)

## Features

### Data Preprocessing
- **Numerical Feature Engineering:**
  - Senior citizen indicator (age >= 60)
  - Positive balance indicator
  - Log transformation of balance
  - Total contacts (campaign + previous)
  - First contact indicator
  - Previously contacted indicator

- **Categorical Feature Engineering:**
  - Age group categorization (Young, Middle-aged, Senior)
  - High success month indicator
  - High success job indicator
  - Loan count and loan presence indicators
  - Combined previous outcome and contact type features

- **Data Scaling & Encoding:**
  - StandardScaler for numerical features
  - Label encoding for categorical features

### Model Features
- **Deep Neural Network:**
  - 4-layer architecture with batch normalization
  - Dropout regularization
  - Class weight balancing
  - Early stopping

- **XGBoost:**
  - Hyperparameter optimization using RandomizedSearchCV
  - Scale positive weight for class imbalance
  - Early stopping with validation monitoring

- **LightGBM:**
  - Hyperparameter tuning
  - Built-in class imbalance handling
  - Efficient gradient boosting

## Installation

### Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow xgboost lightgbm
```

### Optional (for Jupyter notebook)
```bash
pip install jupyter
```

## Usage

### Running the Complete Pipeline
```python
# 1. Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 2. Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 3. Run the preprocessing and modeling pipeline
# (Follow the notebook cells sequentially)
```

### Key Steps

1. **Data Loading and Exploration**
   ```python
   # Load datasets
   train = pd.read_csv('train.csv')
   test = pd.read_csv('test.csv')
   
   # Basic exploration
   train.info()
   train.describe(include='all')
   ```

2. **Feature Engineering**
   ```python
   # Create new numerical features
   train_num_features['is_senior'] = (train_num_features['age'] >= 60).astype(int)
   train_num_features['positive_balance'] = (train_num_features['balance'] > 0).astype(int)
   # ... (additional features)
   ```

3. **Model Training**
   ```python
   # DNN Model
   model_dnn = Sequential([
       layers.Dense(256, input_shape=(input_dim,), kernel_regularizer=regularizers.l2(0.001)),
       layers.BatchNormalization(),
       layers.Activation('relu'),
       layers.Dropout(0.5),
       # ... (additional layers)
   ])
   ```

## Model Architecture

### Deep Neural Network
- **Input Layer:** Variable size based on features
- **Hidden Layers:** 
  - Layer 1: 256 neurons + BatchNorm + ReLU + 50% Dropout
  - Layer 2: 128 neurons + BatchNorm + ReLU + 40% Dropout
  - Layer 3: 64 neurons + BatchNorm + ReLU + 30% Dropout
- **Output Layer:** 1 neuron + Sigmoid activation
- **Regularization:** L2 regularization (0.001), Batch normalization, Dropout
- **Optimization:** Adam optimizer with learning rate 0.001

### XGBoost Configuration
```python
best_params = {
    'max_depth': 7,
    'min_child_weight': 8,
    'subsample': 0.99,
    'colsample_bytree': 0.61,
    'learning_rate': 0.06,
    'n_estimators': 847,
    'gamma': 0.48,
    'reg_alpha': 6.22,
    'reg_lambda': 4.94
}
```

### LightGBM Configuration
```python
best_params_lgb = {
    'max_depth': 14,
    'num_leaves': 290,
    'min_child_samples': 62,
    'subsample': 0.81,
    'colsample_bytree': 0.72,
    'learning_rate': 0.10,
    'n_estimators': 158,
    'reg_alpha': 6.22,
    'reg_lambda': 0.71
}
```

## Results

### Model Performance (Validation ROC-AUC)

| Model | ROC-AUC Score |
|-------|---------------|
| Deep Neural Network | ~0.9200 |
| XGBoost (default) | 0.9671 |
| XGBoost (tuned) | **0.9679** |
| LightGBM (default) | 0.9549 |
| LightGBM (tuned) | 0.9600 |

### Key Insights
- XGBoost with hyperparameter tuning achieved the best performance
- All models showed excellent performance (>0.92 ROC-AUC)
- Feature engineering significantly improved model performance
- Class imbalance handling was crucial for optimal results

## File Structure

```
project/
│
├── classification-with-bank-data-dnn.ipynb  # Main notebook
├── train.csv                                # Training dataset
├── test.csv                                 # Test dataset
├── sample_submission.csv                    # Submission format
├── README.md                               # This file
│
└── outputs/
    ├── model_weights/                      # Saved model weights
    ├── predictions/                        # Model predictions
    └── visualizations/                     # Generated plots
```

## Key Visualizations

The notebook includes comprehensive visualizations:
- **Exploratory Data Analysis:** Distribution plots, correlation heatmaps
- **Feature Analysis:** KDE plots, box plots for numerical features
- **Model Performance:** ROC curves, training history plots
- **Classification Reports:** Precision, recall, F1-scores

## Best Practices Implemented

1. **Data Preprocessing:**
   - Comprehensive feature engineering
   - Proper train/validation/test splits
   - Standardization and encoding

2. **Model Development:**
   - Cross-validation for hyperparameter tuning
   - Class imbalance handling
   - Regularization techniques
   - Early stopping to prevent overfitting

3. **Evaluation:**
   - Multiple metrics (ROC-AUC, Precision, Recall, F1)
   - Validation on unseen data
   - Visual performance analysis

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset sourced from Kaggle Playground Series
- TensorFlow/Keras for deep learning implementation
- XGBoost and LightGBM teams for excellent gradient boosting libraries
- Scikit-learn for preprocessing and evaluation tools
