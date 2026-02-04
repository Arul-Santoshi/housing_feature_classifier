# King County Housing Price Prediction

A linear regression model to predict house prices in King County, Seattle area using the King County Housing Dataset.

## Overview

This project builds a machine learning model that predicts house prices based on various features such as square footage, number of bedrooms/bathrooms, location, and property condition. The model uses linear regression and achieves an R-squared score of approximately 70% on the test set.

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

### Test Set Metrics

| Metric | Value |
|--------|-------|
| **R-squared (R2)** | 0.6998 |
| **Mean Absolute Error (MAE)** | $127,474 |
| **Root Mean Squared Error (RMSE)** | $213,017 |
| **Mean Squared Error (MSE)** | $45,376,406,090 |

### Interpretation
- The model explains **70.0%** of the variance in house prices
- On average, predictions are off by approximately **$127,474**
- Good generalization with minimal overfitting (train/test R2 difference: -0.0018)

### Prediction Accuracy
- **29.1%** of predictions within 10% of actual price
- **51.8%** of predictions within 20% of actual price
- **68.6%** of predictions within 30% of actual price

## Key Findings

### Most Influential Features (by coefficient magnitude)

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
|-- housing_price_prediction.py  # Main prediction script
|-- requirements.txt              # Python dependencies
|-- README.md                     # This file
|-- data/
|   |-- kc_house_data.csv        # Downloaded dataset
|-- output/
    |-- price_distribution.png    # Price distribution histogram
    |-- actual_vs_predicted.png   # Actual vs predicted scatter plot
    |-- residuals_plot.png        # Residuals analysis
    |-- feature_importance.png    # Feature coefficients
    |-- prediction_results.csv    # Detailed predictions
    |-- model_metrics.csv         # Performance metrics
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

3. **Run the model**
   ```bash
   python housing_price_prediction.py
   ```

   The script will:
   - Download the dataset automatically (or use local file if available)
   - Perform exploratory data analysis
   - Preprocess the data (encoding, scaling)
   - Train the linear regression model
   - Generate evaluation metrics and visualizations
   - Save results to the `output/` directory

### Using Your Own Data

To use a different dataset:
1. Place your CSV file in the `data/` directory named `kc_house_data.csv`
2. Ensure it has the same column names as the original dataset
3. Run the script

## Visualizations

### Price Distribution
Shows the distribution of house prices, including a log-transformed view for better visualization of the right-skewed data.

### Actual vs Predicted
Scatter plot comparing actual prices to predicted prices, with a perfect prediction line for reference.

### Residuals Plot
- Left: Residuals vs predicted values (checking for heteroscedasticity)
- Right: Distribution of residuals (checking for normality)

### Feature Importance
Horizontal bar chart showing the coefficient values for each feature, indicating their relative importance and direction of influence.

## Limitations

1. **Linear Assumptions**: Linear regression assumes a linear relationship between features and target, which may not capture complex interactions
2. **Feature Engineering**: Additional derived features (e.g., price per sqft, age of house) could improve performance
3. **Model Complexity**: More complex models (Random Forest, XGBoost, Neural Networks) may achieve better accuracy
4. **Temporal Effects**: The model doesn't account for time-series effects in housing prices

## Future Improvements

- Implement polynomial features for capturing non-linear relationships
- Try ensemble methods (Random Forest, Gradient Boosting)
- Add cross-validation for more robust evaluation
- Feature engineering (house age, renovation flag, location clusters)
- Handle outliers more sophisticatedly
- Add log transformation of the target variable

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

## License

This project is provided under the MIT License. The dataset is publicly available on Kaggle.
