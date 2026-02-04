"""
King County Housing Price Prediction Model
==========================================
A linear regression model to predict house prices in King County, Seattle area.

This script performs:
- Data loading and exploratory data analysis (EDA)
- Feature selection and preprocessing
- Model training using Linear Regression
- Model evaluation with multiple metrics
- Visualization of results

Author: Housing Feature Classifier Project
Dataset: King County House Sales (21,613 houses, 21 features)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory for visualizations and results
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """
    Load the King County Housing Dataset.

    The dataset can be loaded from:
    1. Local CSV file (if exists)
    2. Direct URL download

    Returns:
        pd.DataFrame: The loaded dataset
    """
    # Check for local file first
    local_paths = [
        'data/kc_house_data.csv',
        'kc_house_data.csv',
        'data/house_data.csv'
    ]

    for path in local_paths:
        if os.path.exists(path):
            print(f"Loading data from local file: {path}")
            return pd.read_csv(path)

    # Download from URL if local file not found
    # King County Housing Dataset URLs (multiple fallback options)
    urls = [
        "https://raw.githubusercontent.com/basilhan/datasets/master/kc-house-data.csv",
        "https://raw.githubusercontent.com/andypeng93/King_County_Housing_Prices/master/kc_house_data.csv",
        "https://raw.githubusercontent.com/danieloselu3/Linear-Regression-Project/main/data/kc_house_data.csv",
    ]

    for url in urls:
        print(f"Attempting to download from: {url}")
        try:
            df = pd.read_csv(url)
            # Save locally for future use
            os.makedirs('data', exist_ok=True)
            df.to_csv('data/kc_house_data.csv', index=False)
            print("Dataset downloaded and saved to data/kc_house_data.csv")
            return df
        except Exception as e:
            print(f"Failed: {e}")
            continue

    # If all URLs fail, provide manual download instructions
    print("\nError: Could not download the dataset from any source.")
    print("\nPlease download the King County Housing Dataset manually:")
    print("  1. Go to: https://www.kaggle.com/harlfoxem/housesalesprediction")
    print("  2. Download the dataset")
    print("  3. Place 'kc_house_data.csv' in the 'data/' directory")
    raise Exception("Could not download dataset")


def exploratory_data_analysis(df):
    """
    Perform exploratory data analysis on the dataset.

    Args:
        df: The housing dataset DataFrame

    Returns:
        None (prints analysis and saves plots)
    """
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)

    # 1. Dataset shape
    print(f"\n1. DATASET SHAPE")
    print(f"   Rows: {df.shape[0]:,}")
    print(f"   Columns: {df.shape[1]}")

    # 2. First few rows
    print(f"\n2. FIRST 5 ROWS OF THE DATASET")
    print("-"*60)
    print(df.head().to_string())

    # 3. Column names and data types
    print(f"\n3. COLUMN INFORMATION")
    print("-"*60)
    print(df.dtypes.to_string())

    # 4. Summary statistics
    print(f"\n4. SUMMARY STATISTICS")
    print("-"*60)
    print(df.describe().to_string())

    # 5. Missing values
    print(f"\n5. MISSING VALUES")
    print("-"*60)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0].to_string()
          if missing.sum() > 0 else "   No missing values found!")

    # 6. Price distribution visualization
    print(f"\n6. CREATING PRICE DISTRIBUTION VISUALIZATION...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of prices
    axes[0].hist(df['price'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Price ($)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of House Prices', fontsize=14, fontweight='bold')
    axes[0].axvline(df['price'].mean(), color='red', linestyle='--',
                    label=f'Mean: ${df["price"].mean():,.0f}')
    axes[0].axvline(df['price'].median(), color='green', linestyle='--',
                    label=f'Median: ${df["price"].median():,.0f}')
    axes[0].legend()

    # Log-transformed histogram (better visualization for skewed data)
    axes[1].hist(np.log1p(df['price']), bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[1].set_xlabel('Log(Price + 1)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Log-Transformed Prices', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'price_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/price_distribution.png")

    # 7. Price statistics
    print(f"\n7. PRICE STATISTICS")
    print("-"*60)
    print(f"   Min Price:     ${df['price'].min():>15,.0f}")
    print(f"   Max Price:     ${df['price'].max():>15,.0f}")
    print(f"   Mean Price:    ${df['price'].mean():>15,.0f}")
    print(f"   Median Price:  ${df['price'].median():>15,.0f}")
    print(f"   Std Dev:       ${df['price'].std():>15,.0f}")


def select_and_preprocess_features(df):
    """
    Select key features and preprocess the data.

    Args:
        df: The housing dataset DataFrame

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    print("\n" + "="*60)
    print("FEATURE SELECTION & PREPROCESSING")
    print("="*60)

    # Define features to use
    selected_features = [
        'bedrooms',
        'bathrooms',
        'sqft_living',
        'sqft_lot',
        'floors',
        'waterfront',
        'view',
        'condition',
        'grade',
        'yr_built',
        'yr_renovated',
        'zipcode',
        'lat',
        'long'
    ]

    target = 'price'

    print(f"\n1. SELECTED FEATURES ({len(selected_features)} features)")
    print("-"*60)
    for i, feat in enumerate(selected_features, 1):
        print(f"   {i:2d}. {feat}")
    print(f"\n   Target Variable: {target}")

    # Create a copy with selected features
    features_to_keep = selected_features + [target]
    df_selected = df[features_to_keep].copy()

    # 2. Handle missing values
    print(f"\n2. HANDLING MISSING VALUES")
    print("-"*60)
    initial_rows = len(df_selected)
    missing_before = df_selected.isnull().sum().sum()

    if missing_before > 0:
        print(f"   Missing values found: {missing_before}")
        df_selected = df_selected.dropna()
        print(f"   Rows before: {initial_rows:,}")
        print(f"   Rows after:  {len(df_selected):,}")
        print(f"   Rows dropped: {initial_rows - len(df_selected):,}")
    else:
        print("   No missing values in selected features.")

    # 3. Encode categorical variables
    print(f"\n3. ENCODING CATEGORICAL VARIABLES")
    print("-"*60)

    # Waterfront is already binary (0/1)
    print(f"   waterfront: Already encoded (0/1)")
    print(f"      - Value counts: {dict(df_selected['waterfront'].value_counts())}")

    # Zipcode encoding using Label Encoder
    le_zipcode = LabelEncoder()
    df_selected['zipcode_encoded'] = le_zipcode.fit_transform(df_selected['zipcode'])
    print(f"   zipcode: Label encoded ({len(le_zipcode.classes_)} unique values)")

    # Update feature list to use encoded zipcode
    features_for_model = [f for f in selected_features if f != 'zipcode'] + ['zipcode_encoded']

    # 4. Prepare feature matrix and target vector
    X = df_selected[features_for_model].values
    y = df_selected[target].values

    print(f"\n4. FEATURE MATRIX SHAPE: {X.shape}")
    print(f"   TARGET VECTOR SHAPE: {y.shape}")

    # 5. Split data into training and testing sets
    print(f"\n5. SPLITTING DATA (80% train, 20% test)")
    print("-"*60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"   Training set: {X_train.shape[0]:,} samples")
    print(f"   Testing set:  {X_test.shape[0]:,} samples")

    # 6. Standardize/normalize numerical features
    print(f"\n6. STANDARDIZING FEATURES")
    print("-"*60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("   Features standardized using StandardScaler (mean=0, std=1)")
    print("   Scaler fitted on training data only (prevents data leakage)")

    return X_train_scaled, X_test_scaled, y_train, y_test, features_for_model, scaler


def train_model(X_train, y_train):
    """
    Train a linear regression model.

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        LinearRegression: Trained model
    """
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)

    print("\n   Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("   Model training completed!")
    print(f"   Intercept: ${model.intercept_:,.2f}")

    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """
    Evaluate the trained model and display metrics.

    Args:
        model: Trained LinearRegression model
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
        feature_names: List of feature names

    Returns:
        tuple: (y_pred, metrics_dict)
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics for training set
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mse = mean_squared_error(y_train, y_train_pred)

    # Calculate metrics for test set
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mse = mean_squared_error(y_test, y_test_pred)

    print("\n1. TRAINING SET PERFORMANCE")
    print("-"*60)
    print(f"   R² Score:                    {train_r2:.4f}")
    print(f"   Mean Absolute Error (MAE):   ${train_mae:,.2f}")
    print(f"   Root Mean Squared Error:     ${train_rmse:,.2f}")
    print(f"   Mean Squared Error (MSE):    ${train_mse:,.2f}")

    print("\n2. TEST SET PERFORMANCE")
    print("-"*60)
    print(f"   R² Score:                    {test_r2:.4f}")
    print(f"   Mean Absolute Error (MAE):   ${test_mae:,.2f}")
    print(f"   Root Mean Squared Error:     ${test_rmse:,.2f}")
    print(f"   Mean Squared Error (MSE):    ${test_mse:,.2f}")

    print("\n3. MODEL INTERPRETATION")
    print("-"*60)
    print(f"   The model explains {test_r2*100:.1f}% of the variance in house prices.")
    print(f"   On average, predictions are off by ${test_mae:,.0f}.")

    # Check for overfitting
    r2_diff = train_r2 - test_r2
    if r2_diff > 0.05:
        print(f"\n   WARNING: Possible overfitting detected!")
        print(f"   R² difference (train - test): {r2_diff:.4f}")
    else:
        print(f"\n   Model shows good generalization (R² diff: {r2_diff:.4f})")

    metrics = {
        'train_r2': train_r2,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_mse': train_mse,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_mse': test_mse
    }

    return y_test_pred, metrics


def create_visualizations(model, y_test, y_pred, feature_names):
    """
    Create and save all visualizations.

    Args:
        model: Trained LinearRegression model
        y_test: Actual test values
        y_pred: Predicted values
        feature_names: List of feature names
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    # Calculate residuals
    residuals = y_test - y_pred

    # 1. Actual vs Predicted scatter plot
    print("\n1. Creating Actual vs Predicted plot...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot with alpha for density visualization
    scatter = ax.scatter(y_test, y_pred, alpha=0.3, s=10, c='steelblue', edgecolors='none')

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Actual Price ($)', fontsize=12)
    ax.set_ylabel('Predicted Price ($)', fontsize=12)
    ax.set_title('Actual vs Predicted House Prices', fontsize=14, fontweight='bold')
    ax.legend()

    # Add R² annotation
    r2 = r2_score(y_test, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/actual_vs_predicted.png")

    # 2. Residuals plot
    print("\n2. Creating Residuals plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, c='coral', edgecolors='none')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[0].set_xlabel('Predicted Price ($)', fontsize=12)
    axes[0].set_ylabel('Residuals ($)', fontsize=12)
    axes[0].set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')

    # Add horizontal lines for standard deviations
    std_residual = np.std(residuals)
    axes[0].axhline(y=std_residual, color='red', linestyle='--', alpha=0.5,
                    label=f'+1 Std: ${std_residual:,.0f}')
    axes[0].axhline(y=-std_residual, color='red', linestyle='--', alpha=0.5,
                    label=f'-1 Std: ${-std_residual:,.0f}')
    axes[0].legend()

    # Residuals histogram
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='teal')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals ($)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')

    # Add statistics
    axes[1].text(0.95, 0.95, f'Mean: ${np.mean(residuals):,.0f}\nStd: ${np.std(residuals):,.0f}',
                 transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'residuals_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/residuals_plot.png")

    # 3. Feature importance (coefficients) plot
    print("\n3. Creating Feature Importance (Coefficients) plot...")

    # Get coefficients and create DataFrame
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })

    # Sort by absolute coefficient value
    coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=True)

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
    bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7, edgecolor='black')

    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Coefficient Value (Standardized)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Feature Importance (Linear Regression Coefficients)', fontsize=14, fontweight='bold')

    # Add value labels on bars
    for bar, coef in zip(bars, coef_df['Coefficient']):
        width = bar.get_width()
        ax.text(width + (0.01 * max(abs(coef_df['Coefficient']))), bar.get_y() + bar.get_height()/2,
                f'{coef:,.0f}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/feature_importance.png")

    # Print feature importance summary
    print("\n4. FEATURE IMPORTANCE SUMMARY (sorted by absolute impact)")
    print("-"*60)
    coef_df_sorted = coef_df.sort_values('Abs_Coefficient', ascending=False)
    for _, row in coef_df_sorted.iterrows():
        direction = "+" if row['Coefficient'] > 0 else "-"
        print(f"   {row['Feature']:20s}: {direction}{abs(row['Coefficient']):>12,.2f}")

    return coef_df


def save_results(y_test, y_pred, metrics):
    """
    Save prediction results to a CSV file.

    Args:
        y_test: Actual test values
        y_pred: Predicted values
        metrics: Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    # Calculate residuals and error percentages
    residuals = y_test - y_pred
    error_percentage = np.abs(residuals) / y_test * 100

    # Create results DataFrame
    results_df = pd.DataFrame({
        'actual_price': y_test,
        'predicted_price': np.round(y_pred, 2),
        'residual': np.round(residuals, 2),
        'error_percentage': np.round(error_percentage, 2)
    })

    # Save to CSV
    results_path = os.path.join(OUTPUT_DIR, 'prediction_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n1. Prediction results saved to: {results_path}")
    print(f"   Total records: {len(results_df):,}")

    # Save metrics summary
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(OUTPUT_DIR, 'model_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n2. Model metrics saved to: {metrics_path}")

    # Print results summary
    print(f"\n3. RESULTS SUMMARY")
    print("-"*60)
    print(f"   Average Error %:    {error_percentage.mean():.2f}%")
    print(f"   Median Error %:     {np.median(error_percentage):.2f}%")
    print(f"   Min Error %:        {error_percentage.min():.2f}%")
    print(f"   Max Error %:        {error_percentage.max():.2f}%")

    # Accuracy buckets
    within_10_pct = (error_percentage <= 10).sum() / len(error_percentage) * 100
    within_20_pct = (error_percentage <= 20).sum() / len(error_percentage) * 100
    within_30_pct = (error_percentage <= 30).sum() / len(error_percentage) * 100

    print(f"\n   PREDICTION ACCURACY BREAKDOWN:")
    print(f"   Within 10% of actual: {within_10_pct:.1f}%")
    print(f"   Within 20% of actual: {within_20_pct:.1f}%")
    print(f"   Within 30% of actual: {within_30_pct:.1f}%")

    return results_df


def print_final_summary(metrics, coef_df):
    """
    Print a final summary of the model performance.

    Args:
        metrics: Dictionary of evaluation metrics
        coef_df: DataFrame with feature coefficients
    """
    print("\n" + "="*60)
    print("FINAL MODEL SUMMARY")
    print("="*60)

    print("""
    MODEL: Linear Regression
    DATASET: King County Housing Dataset

    PERFORMANCE METRICS (Test Set):
    """)
    print(f"    R² Score:                    {metrics['test_r2']:.4f}")
    print(f"    Mean Absolute Error (MAE):   ${metrics['test_mae']:,.2f}")
    print(f"    Root Mean Squared Error:     ${metrics['test_rmse']:,.2f}")
    print(f"    Mean Squared Error (MSE):    ${metrics['test_mse']:,.2f}")

    # Top 3 most important features
    top_features = coef_df.nlargest(3, 'Abs_Coefficient')
    print("\n    TOP 3 MOST INFLUENTIAL FEATURES:")
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"    {i}. {row['Feature']} (coefficient: {row['Coefficient']:,.0f})")

    print("""
    OUTPUT FILES:
    - output/price_distribution.png    : Price distribution histogram
    - output/actual_vs_predicted.png   : Actual vs predicted scatter plot
    - output/residuals_plot.png        : Residuals analysis plots
    - output/feature_importance.png    : Feature coefficients bar chart
    - output/prediction_results.csv    : Detailed prediction results
    - output/model_metrics.csv         : Model evaluation metrics
    """)

    print("="*60)
    print("MODEL TRAINING AND EVALUATION COMPLETE!")
    print("="*60)


def main():
    """
    Main function to run the entire pipeline.
    """
    print("\n" + "="*60)
    print("KING COUNTY HOUSING PRICE PREDICTION MODEL")
    print("Using Linear Regression")
    print("="*60)

    # Step 1: Load data
    df = load_data()

    # Step 2: Exploratory Data Analysis
    exploratory_data_analysis(df)

    # Step 3: Feature Selection & Preprocessing
    X_train, X_test, y_train, y_test, feature_names, scaler = select_and_preprocess_features(df)

    # Step 4: Train model
    model = train_model(X_train, y_train)

    # Step 5: Evaluate model
    y_pred, metrics = evaluate_model(model, X_train, X_test, y_train, y_test, feature_names)

    # Step 6: Create visualizations
    coef_df = create_visualizations(model, y_test, y_pred, feature_names)

    # Step 7: Save results
    save_results(y_test, y_pred, metrics)

    # Step 8: Print final summary
    print_final_summary(metrics, coef_df)

    return model, metrics, coef_df


if __name__ == "__main__":
    main()
