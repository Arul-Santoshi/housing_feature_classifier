"""
King County Housing Price Prediction Model - Gradient Boosting (XGBoost)
=========================================================================
A Gradient Boosting regression model using XGBoost to predict house prices
in King County, Seattle area.

This script performs:
- Data loading and preprocessing (consistent with other models)
- Feature selection and standardization
- Model training using XGBoost Regressor
- Model evaluation with multiple metrics
- Visualization of results including feature importance
- Comparison with Linear Regression and Random Forest models

Gradient Boosting vs Random Forest:
- Random Forest: Builds trees independently in parallel, then averages predictions
- Gradient Boosting: Builds trees sequentially, each correcting errors of previous trees
- XGBoost adds regularization to prevent overfitting and is highly optimized

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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Try to import XGBoost, provide helpful message if not installed
try:
    import xgboost as xgb
except ImportError:
    print("XGBoost not installed. Please install with: pip install xgboost")
    print("Or: conda install -c conda-forge xgboost")
    raise

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory for visualizations and results
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

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
    urls = [
        "https://raw.githubusercontent.com/basilhan/datasets/master/kc-house-data.csv",
        "https://raw.githubusercontent.com/andypeng93/King_County_Housing_Prices/master/kc_house_data.csv",
        "https://raw.githubusercontent.com/danieloselu3/Linear-Regression-Project/main/data/kc_house_data.csv",
    ]

    for url in urls:
        print(f"Attempting to download from: {url}")
        try:
            df = pd.read_csv(url)
            os.makedirs('data', exist_ok=True)
            df.to_csv('data/kc_house_data.csv', index=False)
            print("Dataset downloaded and saved to data/kc_house_data.csv")
            return df
        except Exception as e:
            print(f"Failed: {e}")
            continue

    print("\nError: Could not download the dataset from any source.")
    print("\nPlease download the King County Housing Dataset manually:")
    print("  1. Go to: https://www.kaggle.com/harlfoxem/housesalesprediction")
    print("  2. Download the dataset")
    print("  3. Place 'kc_house_data.csv' in the 'data/' directory")
    raise Exception("Could not download dataset")


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def select_and_preprocess_features(df):
    """
    Select key features and preprocess the data.
    Uses the same preprocessing as Linear Regression and Random Forest for consistency.

    Args:
        df: The housing dataset DataFrame

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    print("\n" + "="*60)
    print("FEATURE SELECTION & PREPROCESSING")
    print("="*60)

    # Define features (same as other models for fair comparison)
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

    # Handle missing values
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

    # Encode categorical variables
    print(f"\n3. ENCODING CATEGORICAL VARIABLES")
    print("-"*60)
    print(f"   waterfront: Already encoded (0/1)")
    print(f"      - Value counts: {dict(df_selected['waterfront'].value_counts())}")

    # Zipcode encoding using Label Encoder
    le_zipcode = LabelEncoder()
    df_selected['zipcode_encoded'] = le_zipcode.fit_transform(df_selected['zipcode'])
    print(f"   zipcode: Label encoded ({len(le_zipcode.classes_)} unique values)")

    # Update feature list
    features_for_model = [f for f in selected_features if f != 'zipcode'] + ['zipcode_encoded']

    # Prepare feature matrix and target vector
    X = df_selected[features_for_model].values
    y = df_selected[target].values

    print(f"\n4. FEATURE MATRIX SHAPE: {X.shape}")
    print(f"   TARGET VECTOR SHAPE: {y.shape}")

    # Split data (same random_state for consistency)
    print(f"\n5. SPLITTING DATA (80% train, 20% test)")
    print("-"*60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"   Training set: {X_train.shape[0]:,} samples")
    print(f"   Testing set:  {X_test.shape[0]:,} samples")

    # Standardize features
    print(f"\n6. STANDARDIZING FEATURES")
    print("-"*60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("   Features standardized using StandardScaler (mean=0, std=1)")
    print("   Scaler fitted on training data only (prevents data leakage)")

    return X_train_scaled, X_test_scaled, y_train, y_test, features_for_model, scaler


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_xgboost(X_train, y_train):
    """
    Train an XGBoost regression model.

    XGBoost Parameters Explained:
    - n_estimators: Number of boosting rounds (trees to build sequentially)
    - max_depth: Maximum depth of each tree (controls complexity)
    - learning_rate: Step size shrinkage to prevent overfitting (lower = more conservative)
    - subsample: Fraction of samples used per tree (adds randomness)
    - colsample_bytree: Fraction of features used per tree
    - reg_alpha: L1 regularization (Lasso - feature selection)
    - reg_lambda: L2 regularization (Ridge - prevents large weights)

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        xgb.XGBRegressor: Trained model
    """
    print("\n" + "="*60)
    print("MODEL TRAINING - XGBOOST GRADIENT BOOSTING")
    print("="*60)

    print("\n   Model Parameters:")
    print("-"*60)
    print("   n_estimators:      200 (number of boosting rounds)")
    print("   max_depth:         6 (tree depth - lower than RF to prevent overfitting)")
    print("   learning_rate:     0.1 (step size shrinkage)")
    print("   subsample:         0.8 (80% of samples per tree)")
    print("   colsample_bytree:  0.8 (80% of features per tree)")
    print("   reg_alpha:         0.1 (L1 regularization)")
    print("   reg_lambda:        1.0 (L2 regularization)")
    print("   random_state:      42 (for reproducibility)")

    print("\n   Training XGBoost Regressor...")

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,        # L1 regularization
        reg_lambda=1.0,       # L2 regularization
        random_state=42,
        n_jobs=-1,            # Use all CPU cores
        verbosity=0           # Suppress XGBoost warnings
    )

    model.fit(X_train, y_train)

    print("   Model training completed!")
    print(f"   Number of boosting rounds: {model.n_estimators}")
    print(f"   Number of features: {model.n_features_in_}")

    return model


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """
    Evaluate the trained model and display metrics.

    Args:
        model: Trained XGBRegressor model
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
        print(f"\n   NOTE: Some overfitting detected (R² diff: {r2_diff:.4f})")
        print(f"   XGBoost's regularization helps control this.")
    else:
        print(f"\n   Excellent generalization! (R² diff: {r2_diff:.4f})")
        print(f"   XGBoost's regularization is working well.")

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


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def create_visualizations(model, y_test, y_pred, feature_names):
    """
    Create and save all visualizations.

    Args:
        model: Trained XGBRegressor model
        y_test: Actual test values
        y_pred: Predicted values
        feature_names: List of feature names

    Returns:
        pd.DataFrame: Feature importance DataFrame
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    residuals = y_test - y_pred

    # 1. Actual vs Predicted scatter plot
    print("\n1. Creating Actual vs Predicted plot...")

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(y_test, y_pred, alpha=0.3, s=10, c='darkorange', edgecolors='none')

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Actual Price ($)', fontsize=12)
    ax.set_ylabel('Predicted Price ($)', fontsize=12)
    ax.set_title('Actual vs Predicted House Prices (XGBoost)', fontsize=14, fontweight='bold')
    ax.legend()

    r2 = r2_score(y_test, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/xgb_actual_vs_predicted.png")

    # 2. Residuals plot
    print("\n2. Creating Residuals plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, c='purple', edgecolors='none')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[0].set_xlabel('Predicted Price ($)', fontsize=12)
    axes[0].set_ylabel('Residuals ($)', fontsize=12)
    axes[0].set_title('Residuals vs Predicted Values (XGBoost)', fontsize=14, fontweight='bold')

    std_residual = np.std(residuals)
    axes[0].axhline(y=std_residual, color='red', linestyle='--', alpha=0.5,
                    label=f'+1 Std: ${std_residual:,.0f}')
    axes[0].axhline(y=-std_residual, color='red', linestyle='--', alpha=0.5,
                    label=f'-1 Std: ${-std_residual:,.0f}')
    axes[0].legend()

    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='darkorange')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals ($)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Residuals (XGBoost)', fontsize=14, fontweight='bold')

    axes[1].text(0.95, 0.95, f'Mean: ${np.mean(residuals):,.0f}\nStd: ${np.std(residuals):,.0f}',
                 transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_residuals_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/xgb_residuals_plot.png")

    # 3. Feature importance plot
    print("\n3. Creating Feature Importance plot...")

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })

    importance_df = importance_df.sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(importance_df)))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'],
                   color=colors, alpha=0.8, edgecolor='black')

    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Feature Importance (XGBoost)', fontsize=14, fontweight='bold')

    for bar, imp in zip(bars, importance_df['Importance']):
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/xgb_feature_importance.png")

    # Print feature importance summary
    print("\n4. FEATURE IMPORTANCE SUMMARY (sorted by importance)")
    print("-"*60)
    importance_df_sorted = importance_df.sort_values('Importance', ascending=False)
    for _, row in importance_df_sorted.iterrows():
        pct = row['Importance'] * 100
        print(f"   {row['Feature']:20s}: {row['Importance']:.4f} ({pct:.2f}%)")

    return importance_df


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def load_other_model_metrics():
    """
    Load metrics from Linear Regression and Random Forest models.

    Returns:
        tuple: (lr_metrics, rf_metrics)
    """
    lr_metrics = None
    rf_metrics = None

    # Try to load Linear Regression metrics
    lr_path = os.path.join(OUTPUT_DIR, 'model_metrics.csv')
    try:
        if os.path.exists(lr_path):
            lr_df = pd.read_csv(lr_path)
            lr_metrics = {
                'test_r2': lr_df['test_r2'].values[0],
                'test_mae': lr_df['test_mae'].values[0],
                'test_rmse': lr_df['test_rmse'].values[0],
                'test_mse': lr_df['test_mse'].values[0]
            }
            print("   Loaded Linear Regression metrics from file")
    except Exception:
        pass

    if lr_metrics is None:
        print("   Using hardcoded Linear Regression metrics")
        lr_metrics = {
            'test_r2': 0.6998,
            'test_mae': 127474.11,
            'test_rmse': 213017.42,
            'test_mse': 45376421069.72
        }

    # Try to load Random Forest metrics
    rf_path = os.path.join(OUTPUT_DIR, 'rf_model_metrics.csv')
    try:
        if os.path.exists(rf_path):
            rf_df = pd.read_csv(rf_path)
            rf_metrics = {
                'test_r2': rf_df['test_r2'].values[0],
                'test_mae': rf_df['test_mae'].values[0],
                'test_rmse': rf_df['test_rmse'].values[0],
                'test_mse': rf_df['test_mse'].values[0]
            }
            print("   Loaded Random Forest metrics from file")
    except Exception:
        pass

    if rf_metrics is None:
        print("   Using hardcoded Random Forest metrics")
        rf_metrics = {
            'test_r2': 0.8566,
            'test_mae': 72157.91,
            'test_rmse': 147217.60,
            'test_mse': 21673020617.33
        }

    return lr_metrics, rf_metrics


def create_model_comparison(xgb_metrics, lr_metrics, rf_metrics):
    """
    Create comparison visualization and summary between all three models.

    Args:
        xgb_metrics: XGBoost metrics dictionary
        lr_metrics: Linear Regression metrics dictionary
        rf_metrics: Random Forest metrics dictionary

    Returns:
        dict: Comparison results
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON: LR vs RF vs XGBOOST")
    print("="*60)

    print("\n1. Creating Model Comparison visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models = ['Linear\nRegression', 'Random\nForest', 'XGBoost']
    x_pos = np.arange(len(models))
    colors = ['steelblue', 'forestgreen', 'darkorange']

    # R² Score comparison
    r2_values = [lr_metrics['test_r2'], rf_metrics['test_r2'], xgb_metrics['test_r2']]
    bars1 = axes[0].bar(x_pos, r2_values, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(models)
    axes[0].set_ylim(0, 1)
    for bar, val in zip(bars1, r2_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # MAE comparison
    mae_values = [lr_metrics['test_mae'], rf_metrics['test_mae'], xgb_metrics['test_mae']]
    bars2 = axes[1].bar(x_pos, mae_values, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('Mean Absolute Error ($)', fontsize=12)
    axes[1].set_title('MAE Comparison (lower is better)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(models)
    for bar, val in zip(bars2, mae_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1500,
                     f'${val:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # RMSE comparison
    rmse_values = [lr_metrics['test_rmse'], rf_metrics['test_rmse'], xgb_metrics['test_rmse']]
    bars3 = axes[2].bar(x_pos, rmse_values, color=colors, alpha=0.8, edgecolor='black')
    axes[2].set_ylabel('Root Mean Squared Error ($)', fontsize=12)
    axes[2].set_title('RMSE Comparison (lower is better)', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(models)
    for bar, val in zip(bars3, rmse_values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1500,
                     f'${val:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'all_models_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/all_models_comparison.png")

    # Print comparison table
    print("\n2. MODEL COMPARISON SUMMARY")
    print("-"*75)
    print(f"{'Metric':<20} {'Linear Regression':>18} {'Random Forest':>18} {'XGBoost':>18}")
    print("-"*75)
    print(f"{'R² Score':<20} {lr_metrics['test_r2']:>18.4f} {rf_metrics['test_r2']:>18.4f} {xgb_metrics['test_r2']:>18.4f}")
    print(f"{'MAE ($)':<20} {lr_metrics['test_mae']:>18,.2f} {rf_metrics['test_mae']:>18,.2f} {xgb_metrics['test_mae']:>18,.2f}")
    print(f"{'RMSE ($)':<20} {lr_metrics['test_rmse']:>18,.2f} {rf_metrics['test_rmse']:>18,.2f} {xgb_metrics['test_rmse']:>18,.2f}")
    print(f"{'MSE ($)':<20} {lr_metrics['test_mse']:>18,.2f} {rf_metrics['test_mse']:>18,.2f} {xgb_metrics['test_mse']:>18,.2f}")

    # Calculate improvements
    print("\n3. XGBOOST IMPROVEMENT OVER OTHER MODELS")
    print("-"*75)

    # vs Linear Regression
    lr_r2_imp = ((xgb_metrics['test_r2'] - lr_metrics['test_r2']) / lr_metrics['test_r2']) * 100
    lr_mae_imp = ((lr_metrics['test_mae'] - xgb_metrics['test_mae']) / lr_metrics['test_mae']) * 100
    lr_rmse_imp = ((lr_metrics['test_rmse'] - xgb_metrics['test_rmse']) / lr_metrics['test_rmse']) * 100

    print(f"\n   vs Linear Regression:")
    print(f"      R² Score:  {'+' if lr_r2_imp > 0 else ''}{lr_r2_imp:.2f}%")
    print(f"      MAE:       {'+' if lr_mae_imp > 0 else ''}{lr_mae_imp:.2f}%")
    print(f"      RMSE:      {'+' if lr_rmse_imp > 0 else ''}{lr_rmse_imp:.2f}%")

    # vs Random Forest
    rf_r2_imp = ((xgb_metrics['test_r2'] - rf_metrics['test_r2']) / rf_metrics['test_r2']) * 100
    rf_mae_imp = ((rf_metrics['test_mae'] - xgb_metrics['test_mae']) / rf_metrics['test_mae']) * 100
    rf_rmse_imp = ((rf_metrics['test_rmse'] - xgb_metrics['test_rmse']) / rf_metrics['test_rmse']) * 100

    print(f"\n   vs Random Forest:")
    print(f"      R² Score:  {'+' if rf_r2_imp > 0 else ''}{rf_r2_imp:.2f}%")
    print(f"      MAE:       {'+' if rf_mae_imp > 0 else ''}{rf_mae_imp:.2f}%")
    print(f"      RMSE:      {'+' if rf_rmse_imp > 0 else ''}{rf_rmse_imp:.2f}%")

    # Determine best model
    print("\n4. CONCLUSION")
    print("-"*75)

    all_r2 = [('Linear Regression', lr_metrics['test_r2']),
              ('Random Forest', rf_metrics['test_r2']),
              ('XGBoost', xgb_metrics['test_r2'])]
    best_model = max(all_r2, key=lambda x: x[1])

    print(f"   Best performing model: {best_model[0]} (R² = {best_model[1]:.4f})")

    if best_model[0] == 'XGBoost':
        print(f"   XGBoost outperforms both Linear Regression and Random Forest!")
        print(f"   - Explains {(xgb_metrics['test_r2'] - lr_metrics['test_r2'])*100:.2f}% more variance than LR")
        print(f"   - Explains {(xgb_metrics['test_r2'] - rf_metrics['test_r2'])*100:.2f}% more variance than RF")
    elif best_model[0] == 'Random Forest':
        print(f"   Random Forest still performs best on this dataset.")
        print(f"   Consider hyperparameter tuning for XGBoost.")
    else:
        print(f"   Linear Regression performs best (unusual - check for issues).")

    return {
        'lr_r2_improvement': lr_r2_imp,
        'lr_mae_improvement': lr_mae_imp,
        'lr_rmse_improvement': lr_rmse_imp,
        'rf_r2_improvement': rf_r2_imp,
        'rf_mae_improvement': rf_mae_imp,
        'rf_rmse_improvement': rf_rmse_imp,
        'best_model': best_model[0]
    }


# =============================================================================
# SAVE RESULTS
# =============================================================================

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

    # Save predictions
    results_path = os.path.join(OUTPUT_DIR, 'xgb_prediction_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n1. Prediction results saved to: {results_path}")
    print(f"   Total records: {len(results_df):,}")

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(OUTPUT_DIR, 'xgb_model_metrics.csv')
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


# =============================================================================
# FINAL SUMMARY
# =============================================================================

def print_final_summary(metrics, importance_df, comparison_results):
    """
    Print a final summary of the XGBoost model performance.

    Args:
        metrics: Dictionary of evaluation metrics
        importance_df: DataFrame with feature importances
        comparison_results: Dictionary with comparison metrics
    """
    print("\n" + "="*60)
    print("FINAL MODEL SUMMARY - XGBOOST")
    print("="*60)

    print("""
    MODEL: XGBoost Gradient Boosting Regressor
    DATASET: King County Housing Dataset

    MODEL PARAMETERS:
    - n_estimators: 200
    - max_depth: 6
    - learning_rate: 0.1
    - subsample: 0.8
    - colsample_bytree: 0.8
    - reg_alpha: 0.1 (L1)
    - reg_lambda: 1.0 (L2)
    - random_state: 42

    PERFORMANCE METRICS (Test Set):
    """)
    print(f"    R² Score:                    {metrics['test_r2']:.4f}")
    print(f"    Mean Absolute Error (MAE):   ${metrics['test_mae']:,.2f}")
    print(f"    Root Mean Squared Error:     ${metrics['test_rmse']:,.2f}")
    print(f"    Mean Squared Error (MSE):    ${metrics['test_mse']:,.2f}")

    # Top 3 features
    top_features = importance_df.nlargest(3, 'Importance')
    print("\n    TOP 3 MOST IMPORTANT FEATURES:")
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"    {i}. {row['Feature']} (importance: {row['Importance']:.4f})")

    print(f"""
    COMPARISON WITH OTHER MODELS:
    - vs Linear Regression: R² {comparison_results['lr_r2_improvement']:+.2f}%, MAE {comparison_results['lr_mae_improvement']:+.2f}%
    - vs Random Forest:     R² {comparison_results['rf_r2_improvement']:+.2f}%, MAE {comparison_results['rf_mae_improvement']:+.2f}%

    BEST MODEL: {comparison_results['best_model']}
    """)

    print("""    OUTPUT FILES:
    - output/xgb_actual_vs_predicted.png : Actual vs predicted scatter plot
    - output/xgb_residuals_plot.png      : Residuals analysis plots
    - output/xgb_feature_importance.png  : Feature importance bar chart
    - output/all_models_comparison.png   : LR vs RF vs XGBoost comparison
    - output/xgb_prediction_results.csv  : Detailed prediction results
    - output/xgb_model_metrics.csv       : Model evaluation metrics
    """)

    print("="*60)
    print("XGBOOST MODEL TRAINING AND EVALUATION COMPLETE!")
    print("="*60)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function to run the entire XGBoost pipeline.
    """
    print("\n" + "="*60)
    print("KING COUNTY HOUSING PRICE PREDICTION MODEL")
    print("Using XGBoost Gradient Boosting Regressor")
    print("="*60)

    # Step 1: Load data
    df = load_data()

    # Step 2: Feature Selection & Preprocessing
    X_train, X_test, y_train, y_test, feature_names, scaler = select_and_preprocess_features(df)

    # Step 3: Train XGBoost model
    model = train_xgboost(X_train, y_train)

    # Step 4: Evaluate model
    y_pred, xgb_metrics = evaluate_model(model, X_train, X_test, y_train, y_test, feature_names)

    # Step 5: Create visualizations
    importance_df = create_visualizations(model, y_test, y_pred, feature_names)

    # Step 6: Save results
    save_results(y_test, y_pred, xgb_metrics)

    # Step 7: Load other models' metrics and create comparison
    print("\n" + "="*60)
    print("LOADING OTHER MODELS' METRICS FOR COMPARISON")
    print("="*60)
    lr_metrics, rf_metrics = load_other_model_metrics()
    comparison_results = create_model_comparison(xgb_metrics, lr_metrics, rf_metrics)

    # Step 8: Print final summary
    print_final_summary(xgb_metrics, importance_df, comparison_results)

    return model, xgb_metrics, importance_df


if __name__ == "__main__":
    main()
