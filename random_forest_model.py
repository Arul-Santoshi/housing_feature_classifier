"""
King County Housing Price Prediction Model - Random Forest
============================================================
A Random Forest regression model to predict house prices in King County, Seattle area.

This script performs:
- Data loading and preprocessing (same as linear regression)
- Feature selection and standardization
- Model training using Random Forest Regressor
- Model evaluation with multiple metrics
- Visualization of results including feature importance
- Comparison with Linear Regression model

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
from sklearn.ensemble import RandomForestRegressor
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


def select_and_preprocess_features(df):
    """
    Select key features and preprocess the data.
    Uses the same preprocessing as the linear regression model for consistency.

    Args:
        df: The housing dataset DataFrame

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    print("\n" + "="*60)
    print("FEATURE SELECTION & PREPROCESSING")
    print("="*60)

    # Define features to use (same as linear regression)
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

    # 5. Split data into training and testing sets (same random_state as linear regression)
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


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest regression model.

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        RandomForestRegressor: Trained model
    """
    print("\n" + "="*60)
    print("MODEL TRAINING - RANDOM FOREST")
    print("="*60)

    print("\n   Model Parameters:")
    print("-"*60)
    print("   n_estimators:     100 (number of trees)")
    print("   max_depth:        20 (to prevent overfitting)")
    print("   min_samples_split: 5")
    print("   random_state:     42 (for reproducibility)")

    print("\n   Training Random Forest Regressor...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1  # Use all available cores for faster training
    )

    model.fit(X_train, y_train)

    print("   Model training completed!")
    print(f"   Number of trees: {model.n_estimators}")
    print(f"   Number of features: {model.n_features_in_}")

    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """
    Evaluate the trained model and display metrics.

    Args:
        model: Trained RandomForestRegressor model
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
        print(f"\n   NOTE: Some overfitting detected (expected with Random Forest)")
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
        model: Trained RandomForestRegressor model
        y_test: Actual test values
        y_pred: Predicted values
        feature_names: List of feature names

    Returns:
        pd.DataFrame: Feature importance DataFrame
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
    scatter = ax.scatter(y_test, y_pred, alpha=0.3, s=10, c='forestgreen', edgecolors='none')

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Actual Price ($)', fontsize=12)
    ax.set_ylabel('Predicted Price ($)', fontsize=12)
    ax.set_title('Actual vs Predicted House Prices (Random Forest)', fontsize=14, fontweight='bold')
    ax.legend()

    # Add R² annotation
    r2 = r2_score(y_test, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rf_actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/rf_actual_vs_predicted.png")

    # 2. Residuals plot
    print("\n2. Creating Residuals plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, c='darkorange', edgecolors='none')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[0].set_xlabel('Predicted Price ($)', fontsize=12)
    axes[0].set_ylabel('Residuals ($)', fontsize=12)
    axes[0].set_title('Residuals vs Predicted Values (Random Forest)', fontsize=14, fontweight='bold')

    # Add horizontal lines for standard deviations
    std_residual = np.std(residuals)
    axes[0].axhline(y=std_residual, color='red', linestyle='--', alpha=0.5,
                    label=f'+1 Std: ${std_residual:,.0f}')
    axes[0].axhline(y=-std_residual, color='red', linestyle='--', alpha=0.5,
                    label=f'-1 Std: ${-std_residual:,.0f}')
    axes[0].legend()

    # Residuals histogram
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='darkgreen')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals ($)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Residuals (Random Forest)', fontsize=14, fontweight='bold')

    # Add statistics
    axes[1].text(0.95, 0.95, f'Mean: ${np.mean(residuals):,.0f}\nStd: ${np.std(residuals):,.0f}',
                 transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rf_residuals_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/rf_residuals_plot.png")

    # 3. Feature importance plot (using feature_importances_ from Random Forest)
    print("\n3. Creating Feature Importance plot...")

    # Get feature importances from Random Forest
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })

    # Sort by importance value
    importance_df = importance_df.sort_values('Importance', ascending=True)

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(importance_df)))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'],
                   color=colors, alpha=0.8, edgecolor='black')

    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')

    # Add value labels on bars
    for bar, imp in zip(bars, importance_df['Importance']):
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rf_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/rf_feature_importance.png")

    # Print feature importance summary
    print("\n4. FEATURE IMPORTANCE SUMMARY (sorted by importance)")
    print("-"*60)
    importance_df_sorted = importance_df.sort_values('Importance', ascending=False)
    for _, row in importance_df_sorted.iterrows():
        pct = row['Importance'] * 100
        print(f"   {row['Feature']:20s}: {row['Importance']:.4f} ({pct:.2f}%)")

    return importance_df


def load_linear_regression_metrics():
    """
    Load linear regression metrics from the saved CSV file.
    If not found, use hardcoded values from the README.

    Returns:
        dict: Linear regression metrics
    """
    metrics_path = os.path.join(OUTPUT_DIR, 'model_metrics.csv')

    try:
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            return {
                'test_r2': metrics_df['test_r2'].values[0],
                'test_mae': metrics_df['test_mae'].values[0],
                'test_rmse': metrics_df['test_rmse'].values[0],
                'test_mse': metrics_df['test_mse'].values[0]
            }
    except Exception as e:
        print(f"   Could not load metrics file: {e}")

    # Hardcoded values from README (in case file doesn't exist)
    print("   Using hardcoded Linear Regression metrics from documentation")
    return {
        'test_r2': 0.6998,
        'test_mae': 127474.11,
        'test_rmse': 213017.42,
        'test_mse': 45376421069.72
    }


def create_model_comparison(rf_metrics, lr_metrics):
    """
    Create comparison visualization and summary between Linear Regression and Random Forest.

    Args:
        rf_metrics: Random Forest metrics dictionary
        lr_metrics: Linear Regression metrics dictionary
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON: LINEAR REGRESSION vs RANDOM FOREST")
    print("="*60)

    # Create comparison visualization
    print("\n1. Creating Model Comparison visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = ['Linear\nRegression', 'Random\nForest']
    x_pos = np.arange(len(models))

    # Colors for comparison
    colors = ['steelblue', 'forestgreen']

    # R² Score comparison
    r2_values = [lr_metrics['test_r2'], rf_metrics['test_r2']]
    bars1 = axes[0].bar(x_pos, r2_values, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(models)
    axes[0].set_ylim(0, 1)
    for bar, val in zip(bars1, r2_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # MAE comparison
    mae_values = [lr_metrics['test_mae'], rf_metrics['test_mae']]
    bars2 = axes[1].bar(x_pos, mae_values, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('Mean Absolute Error ($)', fontsize=12)
    axes[1].set_title('MAE Comparison (lower is better)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(models)
    for bar, val in zip(bars2, mae_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
                     f'${val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # RMSE comparison
    rmse_values = [lr_metrics['test_rmse'], rf_metrics['test_rmse']]
    bars3 = axes[2].bar(x_pos, rmse_values, color=colors, alpha=0.8, edgecolor='black')
    axes[2].set_ylabel('Root Mean Squared Error ($)', fontsize=12)
    axes[2].set_title('RMSE Comparison (lower is better)', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(models)
    for bar, val in zip(bars3, rmse_values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
                     f'${val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/model_comparison.png")

    # Print comparison table
    print("\n2. MODEL COMPARISON SUMMARY")
    print("-"*60)
    print(f"{'Metric':<30} {'Linear Regression':>20} {'Random Forest':>20}")
    print("-"*60)
    print(f"{'R² Score':<30} {lr_metrics['test_r2']:>20.4f} {rf_metrics['test_r2']:>20.4f}")
    print(f"{'MAE ($)':<30} {lr_metrics['test_mae']:>20,.2f} {rf_metrics['test_mae']:>20,.2f}")
    print(f"{'RMSE ($)':<30} {lr_metrics['test_rmse']:>20,.2f} {rf_metrics['test_rmse']:>20,.2f}")
    print(f"{'MSE ($)':<30} {lr_metrics['test_mse']:>20,.2f} {rf_metrics['test_mse']:>20,.2f}")

    # Calculate improvements
    print("\n3. RANDOM FOREST IMPROVEMENT OVER LINEAR REGRESSION")
    print("-"*60)

    r2_improvement = ((rf_metrics['test_r2'] - lr_metrics['test_r2']) / lr_metrics['test_r2']) * 100
    mae_improvement = ((lr_metrics['test_mae'] - rf_metrics['test_mae']) / lr_metrics['test_mae']) * 100
    rmse_improvement = ((lr_metrics['test_rmse'] - rf_metrics['test_rmse']) / lr_metrics['test_rmse']) * 100

    print(f"   R² Score:  {'+' if r2_improvement > 0 else ''}{r2_improvement:.2f}% {'(better)' if r2_improvement > 0 else '(worse)'}")
    print(f"   MAE:       {'+' if mae_improvement > 0 else ''}{mae_improvement:.2f}% {'(better)' if mae_improvement > 0 else '(worse)'}")
    print(f"   RMSE:      {'+' if rmse_improvement > 0 else ''}{rmse_improvement:.2f}% {'(better)' if rmse_improvement > 0 else '(worse)'}")

    # Overall winner
    print("\n4. CONCLUSION")
    print("-"*60)
    if rf_metrics['test_r2'] > lr_metrics['test_r2']:
        print("   Random Forest outperforms Linear Regression!")
        print(f"   - Explains {(rf_metrics['test_r2'] - lr_metrics['test_r2'])*100:.2f}% more variance")
        print(f"   - Average prediction error reduced by ${lr_metrics['test_mae'] - rf_metrics['test_mae']:,.0f}")
    else:
        print("   Linear Regression performs better for this dataset.")
        print("   Consider: smaller dataset may not benefit from Random Forest complexity.")

    return {
        'r2_improvement': r2_improvement,
        'mae_improvement': mae_improvement,
        'rmse_improvement': rmse_improvement
    }


def save_results(y_test, y_pred, metrics):
    """
    Save prediction results to CSV files.

    Args:
        y_test: Actual test values
        y_pred: Predicted values
        metrics: Dictionary of evaluation metrics

    Returns:
        pd.DataFrame: Results DataFrame
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
    results_path = os.path.join(OUTPUT_DIR, 'rf_prediction_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n1. Prediction results saved to: {results_path}")
    print(f"   Total records: {len(results_df):,}")

    # Save metrics summary
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(OUTPUT_DIR, 'rf_model_metrics.csv')
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


def print_final_summary(metrics, importance_df, comparison_results):
    """
    Print a final summary of the Random Forest model performance.

    Args:
        metrics: Dictionary of evaluation metrics
        importance_df: DataFrame with feature importances
        comparison_results: Dictionary with comparison metrics
    """
    print("\n" + "="*60)
    print("FINAL MODEL SUMMARY - RANDOM FOREST")
    print("="*60)

    print("""
    MODEL: Random Forest Regressor
    DATASET: King County Housing Dataset

    MODEL PARAMETERS:
    - n_estimators: 100
    - max_depth: 20
    - min_samples_split: 5
    - random_state: 42

    PERFORMANCE METRICS (Test Set):
    """)
    print(f"    R² Score:                    {metrics['test_r2']:.4f}")
    print(f"    Mean Absolute Error (MAE):   ${metrics['test_mae']:,.2f}")
    print(f"    Root Mean Squared Error:     ${metrics['test_rmse']:,.2f}")
    print(f"    Mean Squared Error (MSE):    ${metrics['test_mse']:,.2f}")

    # Top 3 most important features
    top_features = importance_df.nlargest(3, 'Importance')
    print("\n    TOP 3 MOST IMPORTANT FEATURES:")
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"    {i}. {row['Feature']} (importance: {row['Importance']:.4f})")

    print(f"""
    COMPARISON WITH LINEAR REGRESSION:
    - R² Improvement:   {comparison_results['r2_improvement']:+.2f}%
    - MAE Improvement:  {comparison_results['mae_improvement']:+.2f}%
    - RMSE Improvement: {comparison_results['rmse_improvement']:+.2f}%
    """)

    print("""    OUTPUT FILES:
    - output/rf_actual_vs_predicted.png : Actual vs predicted scatter plot
    - output/rf_residuals_plot.png      : Residuals analysis plots
    - output/rf_feature_importance.png  : Feature importance bar chart
    - output/model_comparison.png       : Linear Regression vs Random Forest comparison
    - output/rf_prediction_results.csv  : Detailed prediction results
    - output/rf_model_metrics.csv       : Model evaluation metrics
    """)

    print("="*60)
    print("RANDOM FOREST MODEL TRAINING AND EVALUATION COMPLETE!")
    print("="*60)


def main():
    """
    Main function to run the entire Random Forest pipeline.
    """
    print("\n" + "="*60)
    print("KING COUNTY HOUSING PRICE PREDICTION MODEL")
    print("Using Random Forest Regressor")
    print("="*60)

    # Step 1: Load data
    df = load_data()

    # Step 2: Feature Selection & Preprocessing (same as linear regression)
    X_train, X_test, y_train, y_test, feature_names, scaler = select_and_preprocess_features(df)

    # Step 3: Train Random Forest model
    model = train_random_forest(X_train, y_train)

    # Step 4: Evaluate model
    y_pred, rf_metrics = evaluate_model(model, X_train, X_test, y_train, y_test, feature_names)

    # Step 5: Create visualizations
    importance_df = create_visualizations(model, y_test, y_pred, feature_names)

    # Step 6: Save results
    save_results(y_test, y_pred, rf_metrics)

    # Step 7: Load Linear Regression metrics and create comparison
    print("\n" + "="*60)
    print("LOADING LINEAR REGRESSION METRICS FOR COMPARISON")
    print("="*60)
    lr_metrics = load_linear_regression_metrics()
    comparison_results = create_model_comparison(rf_metrics, lr_metrics)

    # Step 8: Print final summary
    print_final_summary(rf_metrics, importance_df, comparison_results)

    return model, rf_metrics, importance_df


if __name__ == "__main__":
    main()
