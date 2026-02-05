# King County Housing Price Prediction

Machine learning models to predict house prices in King County, Seattle area using the King County Housing Dataset.

## Overview

This project builds machine learning models that predict house prices based on various features such as square footage, number of bedrooms/bathrooms, location, and property condition. The project implements three regression models:

1. **Linear Regression** - A baseline model achieving R² = 0.70
2. **Random Forest Regressor** - An ensemble model achieving R² = 0.86
3. **XGBoost Gradient Boosting** - Best performing model achieving R² = 0.87

XGBoost achieves the best performance with an R² of 0.8707, reducing prediction errors by 46% compared to Linear Regression and 5% compared to Random Forest.

## Dataset

**King County House Sales Dataset**
- **Source**: [Kaggle - House Sales in King County, USA](https://www.kaggle.com/harlfoxem/housesalesprediction)
- **Records**: 21,613 houses
- **Features**: 21 variables including price, bedrooms, bathrooms, sqft_living, and more
- **Region**: King County, Washington (Seattle area)
- **Time Period**: May 2014 - May 2015

## Features Used

The model uses the following 14 features:

| Feature | Description |
|---------|-------------|
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| sqft_living | Square footage of living space |
| sqft_lot | Lot size in square feet |
| floors | Number of floors |
| waterfront | Waterfront property (0/1) |
| view | View quality rating (0-4) |
| condition | Condition rating (1-5) |
| grade | Construction quality grade (1-13) |
| yr_built | Year built |
| yr_renovated | Year renovated (0 if never) |
| zipcode | Location zip code |
| lat | Latitude |
| long | Longitude |

## Model Performance

### Model Comparison Summary

| Metric | Linear Regression | Random Forest | XGBoost | Best Improvement |
|--------|------------------|---------------|---------|------------------|
| **R² Score** | 0.6998 | 0.8566 | **0.8707** | +24.41% vs LR |
| **MAE** | $127,474 | $72,158 | **$68,452** | +46.30% vs LR |
| **RMSE** | $213,017 | $147,218 | **$139,832** | +34.36% vs LR |
| **MSE** | $45.4B | $21.7B | **$19.6B** | +56.93% vs LR |

### XGBoost Model (Best Performing)

**Test Set Metrics:**
| Metric | Value |
|--------|-------|
| **R-squared (R²)** | 0.8707 |
| **Mean Absolute Error (MAE)** | $68,451.67 |
| **Root Mean Squared Error (RMSE)** | $139,832.02 |
| **Mean Squared Error (MSE)** | $19,552,993,644 |

**Model Parameters:**
- `n_estimators`: 200 boosting rounds
- `max_depth`: 6 (lower than RF - regularization)
- `learning_rate`: 0.1 (step size shrinkage)
- `subsample`: 0.8 (80% samples per tree)
- `colsample_bytree`: 0.8 (80% features per tree)
- `reg_alpha`: 0.1 (L1 regularization)
- `reg_lambda`: 1.0 (L2 regularization)
- `random_state`: 42

**Interpretation:**
- The model explains **87.1%** of the variance in house prices
- On average, predictions are off by approximately **$68,452**
- Better generalization than Random Forest (train/test R² difference: 0.0943) due to regularization

**Prediction Accuracy:**
- **56.6%** of predictions within 10% of actual price
- **82.6%** of predictions within 20% of actual price
- **92.4%** of predictions within 30% of actual price

### Random Forest Model

**Test Set Metrics:**
| Metric | Value |
|--------|-------|
| **R-squared (R²)** | 0.8566 |
| **Mean Absolute Error (MAE)** | $72,157.91 |
| **Root Mean Squared Error (RMSE)** | $147,217.60 |
| **Mean Squared Error (MSE)** | $21,673,020,617 |

**Model Parameters:**
- `n_estimators`: 100 trees
- `max_depth`: 20 (to prevent overfitting)
- `min_samples_split`: 5
- `random_state`: 42

**Interpretation:**
- The model explains **85.7%** of the variance in house prices
- On average, predictions are off by approximately **$72,158**
- Some overfitting detected (train/test R² difference: 0.1176), which is expected with Random Forest

**Prediction Accuracy:**
- **54.7%** of predictions within 10% of actual price
- **80.8%** of predictions within 20% of actual price
- **90.7%** of predictions within 30% of actual price

### Linear Regression Model (Baseline)

**Test Set Metrics:**
| Metric | Value |
|--------|-------|
| **R-squared (R²)** | 0.6998 |
| **Mean Absolute Error (MAE)** | $127,473.96 |
| **Root Mean Squared Error (RMSE)** | $213,017.38 |
| **Mean Squared Error (MSE)** | $45,376,406,090 |

**Interpretation:**
- The model explains **70.0%** of the variance in house prices
- On average, predictions are off by approximately **$127,474**
- Good generalization with minimal overfitting (train/test R² difference: -0.0018)

**Prediction Accuracy:**
- **29.1%** of predictions within 10% of actual price
- **51.8%** of predictions within 20% of actual price
- **68.6%** of predictions within 30% of actual price

## Key Findings

### XGBoost Feature Importance (Best Model)

The XGBoost model reveals the most predictive features based on gain importance:

| Rank | Feature | Importance | Percentage |
|------|---------|------------|------------|
| 1 | **grade** | 0.3429 | 34.29% |
| 2 | **waterfront** | 0.2123 | 21.23% |
| 3 | **sqft_living** | 0.1601 | 16.01% |
| 4 | **lat** | 0.0772 | 7.72% |
| 5 | **view** | 0.0421 | 4.21% |
| 6 | **long** | 0.0367 | 3.67% |
| 7 | **bathrooms** | 0.0323 | 3.23% |
| 8 | **yr_built** | 0.0244 | 2.44% |
| 9 | **zipcode** | 0.0211 | 2.11% |
| 10 | **yr_renovated** | 0.0143 | 1.43% |
| 11 | **sqft_lot** | 0.0138 | 1.38% |
| 12 | **condition** | 0.0091 | 0.91% |
| 13 | **floors** | 0.0086 | 0.86% |
| 14 | **bedrooms** | 0.0053 | 0.53% |

**Key Insights:**
- **grade** (construction quality) is the most important predictor at 34.29%
- **waterfront** has much higher importance in XGBoost (21.23%) vs Random Forest (3.47%) - XGBoost better captures this binary feature's impact
- **sqft_living** remains crucial at 16.01%
- The top 3 features account for **71.53%** of total feature importance

### Random Forest Feature Importance

| Rank | Feature | Importance | Percentage |
|------|---------|------------|------------|
| 1 | **grade** | 0.3245 | 32.45% |
| 2 | **sqft_living** | 0.2922 | 29.22% |
| 3 | **lat** | 0.1625 | 16.25% |
| 4 | **long** | 0.0690 | 6.90% |
| 5 | **yr_built** | 0.0366 | 3.66% |

### Linear Regression Feature Coefficients

For the Linear Regression model (standardized coefficients showing direction and magnitude):

1. **sqft_living** (+$160,448) - Living space square footage has the strongest positive impact
2. **grade** (+$118,447) - Construction quality significantly affects price
3. **lat** (+$83,237) - Northern locations tend to be more expensive
4. **yr_built** (-$79,980) - Older homes tend to be priced lower
5. **waterfront** (+$47,660) - Waterfront properties command a premium

### Price Distribution
- **Minimum**: $75,000
- **Maximum**: $7,700,000
- **Mean**: $540,088
- **Median**: $450,000

The price distribution is right-skewed, with most houses priced below the mean.

## Project Structure

```
housing_feature_classifier/
├── housing_price_prediction.py   # Linear Regression model script
├── random_forest_model.py        # Random Forest model script
├── gradient_boosting_model.py    # XGBoost model script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── data/
│   └── kc_house_data.csv         # Downloaded dataset
└── output/
    ├── price_distribution.png       # Price distribution histogram
    ├── actual_vs_predicted.png      # Linear Regression: Actual vs predicted
    ├── residuals_plot.png           # Linear Regression: Residuals analysis
    ├── feature_importance.png       # Linear Regression: Feature coefficients
    ├── prediction_results.csv       # Linear Regression: Detailed predictions
    ├── model_metrics.csv            # Linear Regression: Performance metrics
    ├── rf_actual_vs_predicted.png   # Random Forest: Actual vs predicted
    ├── rf_residuals_plot.png        # Random Forest: Residuals analysis
    ├── rf_feature_importance.png    # Random Forest: Feature importance
    ├── rf_prediction_results.csv    # Random Forest: Detailed predictions
    ├── rf_model_metrics.csv         # Random Forest: Performance metrics
    ├── model_comparison.png         # LR vs RF comparison
    ├── xgb_actual_vs_predicted.png  # XGBoost: Actual vs predicted
    ├── xgb_residuals_plot.png       # XGBoost: Residuals analysis
    ├── xgb_feature_importance.png   # XGBoost: Feature importance
    ├── xgb_prediction_results.csv   # XGBoost: Detailed predictions
    ├── xgb_model_metrics.csv        # XGBoost: Performance metrics
    └── all_models_comparison.png    # LR vs RF vs XGBoost comparison
```

## Replication Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd housing_feature_classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Linear Regression model** (baseline)
   ```bash
   python housing_price_prediction.py
   ```

4. **Run the Random Forest model**
   ```bash
   python random_forest_model.py
   ```

5. **Run the XGBoost model** (best performing)
   ```bash
   python gradient_boosting_model.py
   ```

   All scripts will:
   - Download the dataset automatically (or use local file if available)
   - Preprocess the data (encoding, scaling)
   - Train the respective model
   - Generate evaluation metrics and visualizations
   - Save results to the `output/` directory
   - Create model comparison visualizations

### Using Your Own Data

To use a different dataset:
1. Place your CSV file in the `data/` directory named `kc_house_data.csv`
2. Ensure it has the same column names as the original dataset
3. Run the script

## Visualizations

### Price Distribution
Shows the distribution of house prices, including a log-transformed view for better visualization of the right-skewed data.

### Actual vs Predicted
Scatter plot comparing actual prices to predicted prices, with a perfect prediction line for reference. Available for all models:
- `actual_vs_predicted.png` - Linear Regression
- `rf_actual_vs_predicted.png` - Random Forest
- `xgb_actual_vs_predicted.png` - XGBoost

### Residuals Plot
- Left: Residuals vs predicted values (checking for heteroscedasticity)
- Right: Distribution of residuals (checking for normality)

Available for all models: `residuals_plot.png`, `rf_residuals_plot.png`, `xgb_residuals_plot.png`

### Feature Importance
- **Linear Regression** (`feature_importance.png`): Horizontal bar chart showing coefficient values
- **Random Forest** (`rf_feature_importance.png`): Bar chart showing Gini importance scores
- **XGBoost** (`xgb_feature_importance.png`): Bar chart showing gain-based importance scores

### Model Comparison
- `model_comparison.png` - Linear Regression vs Random Forest comparison
- `all_models_comparison.png` - All three models (LR vs RF vs XGBoost) side-by-side comparison across R² Score, MAE, and RMSE

## Limitations

1. **Model Overfitting**: Both Random Forest (train/test R² diff: 0.12) and XGBoost (diff: 0.09) show some overfitting, though XGBoost's regularization helps control this
2. **Feature Engineering**: Additional derived features (e.g., price per sqft, age of house) could improve performance
3. **Temporal Effects**: The models don't account for time-series effects in housing prices
4. **Outlier Sensitivity**: High-value properties (>$2M) may have larger prediction errors
5. **Geographic Granularity**: Zipcode encoding may not capture neighborhood-level price variations

## Future Improvements

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV for all models
- Add cross-validation for more robust evaluation
- Feature engineering (house age, renovation flag, location clusters, price per sqft)
- Try LightGBM and CatBoost for comparison
- Handle outliers more sophisticatedly (robust scaling, trimming)
- Add log transformation of the target variable
- Implement model stacking/blending for ensemble predictions
- Add SHAP values for better model interpretability

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- xgboost >= 1.7.0

## License

This project is provided under the MIT License. The dataset is publicly available on Kaggle.
