# King County Housing Price Prediction

Machine learning models to predict house prices in King County, Seattle area using the King County Housing Dataset.

## Overview

This project builds machine learning models that predict house prices based on various features such as square footage, number of bedrooms/bathrooms, location, and property condition. The project implements two regression models:

1. **Linear Regression** - A baseline model achieving R² = 0.70
2. **Random Forest Regressor** - An ensemble model achieving R² = 0.86

The Random Forest model significantly outperforms Linear Regression, reducing prediction errors by over 43%.

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

| Metric | Linear Regression | Random Forest | Improvement |
|--------|------------------|---------------|-------------|
| **R² Score** | 0.6998 | **0.8566** | +22.40% |
| **MAE** | $127,474 | **$72,158** | +43.39% |
| **RMSE** | $213,017 | **$147,218** | +30.89% |
| **MSE** | $45.4B | **$21.7B** | +52.24% |

### Random Forest Model (Best Performing)

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

### Random Forest Feature Importance

The Random Forest model reveals the most predictive features based on Gini importance:

| Rank | Feature | Importance | Percentage |
|------|---------|------------|------------|
| 1 | **grade** | 0.3245 | 32.45% |
| 2 | **sqft_living** | 0.2922 | 29.22% |
| 3 | **lat** | 0.1625 | 16.25% |
| 4 | **long** | 0.0690 | 6.90% |
| 5 | **yr_built** | 0.0366 | 3.66% |
| 6 | **waterfront** | 0.0347 | 3.47% |
| 7 | **sqft_lot** | 0.0242 | 2.42% |
| 8 | **zipcode** | 0.0187 | 1.87% |
| 9 | **bathrooms** | 0.0133 | 1.33% |
| 10 | **view** | 0.0121 | 1.21% |
| 11 | **bedrooms** | 0.0038 | 0.38% |
| 12 | **condition** | 0.0034 | 0.34% |
| 13 | **floors** | 0.0026 | 0.26% |
| 14 | **yr_renovated** | 0.0025 | 0.25% |

**Key Insights:**
- **grade** (construction quality) is the most important predictor, accounting for 32.45% of the model's decision-making
- **sqft_living** follows closely at 29.22%, confirming that living space size is crucial
- **Location matters**: lat and long together contribute 23.15% of importance
- The top 3 features alone account for **77.92%** of total feature importance

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
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── data/
│   └── kc_house_data.csv         # Downloaded dataset
└── output/
    ├── price_distribution.png    # Price distribution histogram
    ├── actual_vs_predicted.png   # Linear Regression: Actual vs predicted
    ├── residuals_plot.png        # Linear Regression: Residuals analysis
    ├── feature_importance.png    # Linear Regression: Feature coefficients
    ├── prediction_results.csv    # Linear Regression: Detailed predictions
    ├── model_metrics.csv         # Linear Regression: Performance metrics
    ├── rf_actual_vs_predicted.png   # Random Forest: Actual vs predicted
    ├── rf_residuals_plot.png        # Random Forest: Residuals analysis
    ├── rf_feature_importance.png    # Random Forest: Feature importance
    ├── rf_prediction_results.csv    # Random Forest: Detailed predictions
    ├── rf_model_metrics.csv         # Random Forest: Performance metrics
    └── model_comparison.png         # Model comparison visualization
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

3. **Run the Linear Regression model**
   ```bash
   python housing_price_prediction.py
   ```

4. **Run the Random Forest model**
   ```bash
   python random_forest_model.py
   ```

   Both scripts will:
   - Download the dataset automatically (or use local file if available)
   - Preprocess the data (encoding, scaling)
   - Train the respective model
   - Generate evaluation metrics and visualizations
   - Save results to the `output/` directory

   The Random Forest script additionally creates a model comparison visualization.

### Using Your Own Data

To use a different dataset:
1. Place your CSV file in the `data/` directory named `kc_house_data.csv`
2. Ensure it has the same column names as the original dataset
3. Run the script

## Visualizations

### Price Distribution
Shows the distribution of house prices, including a log-transformed view for better visualization of the right-skewed data.

### Actual vs Predicted
Scatter plot comparing actual prices to predicted prices, with a perfect prediction line for reference. Available for both models:
- `actual_vs_predicted.png` - Linear Regression
- `rf_actual_vs_predicted.png` - Random Forest

### Residuals Plot
- Left: Residuals vs predicted values (checking for heteroscedasticity)
- Right: Distribution of residuals (checking for normality)

Available for both models with `residuals_plot.png` and `rf_residuals_plot.png`.

### Feature Importance
- **Linear Regression** (`feature_importance.png`): Horizontal bar chart showing coefficient values
- **Random Forest** (`rf_feature_importance.png`): Bar chart showing Gini importance scores

### Model Comparison
`model_comparison.png` provides a side-by-side comparison of Linear Regression vs Random Forest across three key metrics: R² Score, MAE, and RMSE.

## Limitations

1. **Random Forest Overfitting**: The Random Forest model shows some overfitting (train R² = 0.97 vs test R² = 0.86), though this is expected behavior for ensemble tree models
2. **Feature Engineering**: Additional derived features (e.g., price per sqft, age of house) could improve performance
3. **Temporal Effects**: The models don't account for time-series effects in housing prices
4. **Outlier Sensitivity**: High-value properties (>$2M) may have larger prediction errors

## Future Improvements

- Implement Gradient Boosting (XGBoost, LightGBM) for potential further improvements
- Add cross-validation for more robust evaluation
- Feature engineering (house age, renovation flag, location clusters)
- Handle outliers more sophisticatedly
- Add log transformation of the target variable
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Implement model stacking/blending for ensemble predictions

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

## License

This project is provided under the MIT License. The dataset is publicly available on Kaggle.
