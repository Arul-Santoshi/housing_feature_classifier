"""
XGBoost Hyperparameter Tuning for Housing Price Prediction
===========================================================
This script performs hyperparameter tuning on the XGBoost model to find
the optimal configuration for predicting King County housing prices.

Tuning Methods Available:
1. RandomizedSearchCV - Fast, samples random combinations (default)
2. GridSearchCV - Exhaustive, tests all combinations (slower)

The script will:
- Load and preprocess the data (same as other models)
- Define a hyperparameter search space
- Perform cross-validated search for best parameters
- Train final model with best parameters
- Compare tuned vs untuned performance
- Save results and best parameters

Usage:
    python hyperparameter_tuning.py              # Default: RandomizedSearchCV with 50 iterations
    python hyperparameter_tuning.py --method grid   # GridSearchCV (slower, more thorough)
    python hyperparameter_tuning.py --n-iter 100    # More iterations for better results

Author: Housing Feature Classifier Project
"""

import os
import sys
import time
import json
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer

try:
    import xgboost as xgb
except ImportError:
    print("XGBoost not installed. Please install with: pip install xgboost")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_data():
    """Load the King County Housing Dataset."""
    local_paths = [
        'data/kc_house_data.csv',
        'kc_house_data.csv',
        'data/house_data.csv'
    ]

    for path in local_paths:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            return pd.read_csv(path)

    raise FileNotFoundError(
        "Dataset not found. Please run one of the model scripts first to download the data."
    )


def preprocess_data(df):
    """
    Preprocess data using the same pipeline as other models.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)

    # Same features as other models
    selected_features = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition', 'grade', 'yr_built',
        'yr_renovated', 'zipcode', 'lat', 'long'
    ]
    target = 'price'

    # Select and clean data
    df_selected = df[selected_features + [target]].copy()
    df_selected = df_selected.dropna()

    # Encode zipcode
    le_zipcode = LabelEncoder()
    df_selected['zipcode_encoded'] = le_zipcode.fit_transform(df_selected['zipcode'])

    # Prepare features
    features_for_model = [f for f in selected_features if f != 'zipcode'] + ['zipcode_encoded']
    X = df_selected[features_for_model].values
    y = df_selected[target].values

    print(f"   Dataset size: {len(X):,} samples")
    print(f"   Features: {len(features_for_model)}")

    # Split data (same random_state for consistency)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"   Training set: {len(X_train_scaled):,} samples")
    print(f"   Test set: {len(X_test_scaled):,} samples")

    return X_train_scaled, X_test_scaled, y_train, y_test, features_for_model


# =============================================================================
# HYPERPARAMETER SEARCH SPACE
# =============================================================================

def get_param_distributions():
    """
    Define the hyperparameter search space for RandomizedSearchCV.

    These ranges are based on best practices and typical values that work
    well for housing price prediction tasks.

    Returns:
        dict: Parameter distributions for randomized search
    """
    return {
        # Number of boosting rounds (trees)
        # More trees generally = better performance, but diminishing returns after ~300
        'n_estimators': [100, 150, 200, 250, 300, 400, 500],

        # Maximum tree depth - controls model complexity
        # Lower values = more regularization, less overfitting
        'max_depth': [3, 4, 5, 6, 7, 8, 10],

        # Learning rate (eta) - step size shrinkage
        # Lower values require more trees but can give better results
        'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],

        # Subsample ratio of training instances
        # Lower values add randomness and prevent overfitting
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],

        # Subsample ratio of features when constructing each tree
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],

        # Minimum sum of instance weight needed in a child
        # Higher values = more conservative model
        'min_child_weight': [1, 3, 5, 7, 10],

        # Minimum loss reduction required to make a split
        # Higher values = more conservative
        'gamma': [0, 0.1, 0.2, 0.3, 0.5],

        # L1 regularization term (Lasso)
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],

        # L2 regularization term (Ridge)
        'reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0],
    }


def get_param_grid():
    """
    Define a smaller parameter grid for GridSearchCV.

    GridSearchCV tests ALL combinations, so we use fewer values per parameter
    to keep computation time reasonable.

    Total combinations: 3^9 = 19,683 (with 5-fold CV = 98,415 model fits)
    This is still expensive, so we use a reduced grid.

    Returns:
        dict: Parameter grid for grid search
    """
    return {
        'n_estimators': [200, 300, 400],
        'max_depth': [5, 6, 7],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1.0, 2.0],
    }


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

def run_hyperparameter_search(X_train, y_train, method='random', n_iter=50, cv=5):
    """
    Run hyperparameter search using specified method.

    Args:
        X_train: Training features
        y_train: Training target
        method: 'random' for RandomizedSearchCV, 'grid' for GridSearchCV
        n_iter: Number of iterations for RandomizedSearchCV
        cv: Number of cross-validation folds

    Returns:
        tuple: (best_params, best_score, cv_results, search_time)
    """
    print("\n" + "="*60)
    print(f"HYPERPARAMETER TUNING ({method.upper()})")
    print("="*60)

    # Base model
    base_model = xgb.XGBRegressor(
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    # Custom scorer (negative MSE, sklearn convention)
    scorer = make_scorer(r2_score)

    if method == 'random':
        param_space = get_param_distributions()
        total_combinations = 1
        for values in param_space.values():
            total_combinations *= len(values)

        print(f"\n   Search Method: RandomizedSearchCV")
        print(f"   Parameter space size: {total_combinations:,} possible combinations")
        print(f"   Sampling: {n_iter} random combinations")
        print(f"   Cross-validation folds: {cv}")
        print(f"   Total model fits: {n_iter * cv:,}")

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_space,
            n_iter=n_iter,
            scoring=scorer,
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
    else:  # grid search
        param_grid = get_param_grid()
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)

        print(f"\n   Search Method: GridSearchCV")
        print(f"   Parameter grid size: {total_combinations:,} combinations")
        print(f"   Cross-validation folds: {cv}")
        print(f"   Total model fits: {total_combinations * cv:,}")
        print(f"\n   WARNING: This may take a while...")

        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

    print(f"\n   Starting search...")
    start_time = time.time()
    search.fit(X_train, y_train)
    search_time = time.time() - start_time

    print(f"\n   Search completed in {search_time:.1f} seconds ({search_time/60:.1f} minutes)")

    return search.best_params_, search.best_score_, search.cv_results_, search_time


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_models(X_train, X_test, y_train, y_test, best_params):
    """
    Compare tuned model vs baseline (untuned) model.

    Args:
        X_train, X_test, y_train, y_test: Data splits
        best_params: Best hyperparameters from search

    Returns:
        tuple: (baseline_metrics, tuned_metrics, baseline_model, tuned_model)
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON: BASELINE vs TUNED")
    print("="*60)

    # Baseline model (same as gradient_boosting_model.py)
    baseline_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }

    print("\n1. BASELINE MODEL (Original Parameters)")
    print("-"*60)
    for param, value in baseline_params.items():
        if param not in ['random_state', 'n_jobs', 'verbosity']:
            print(f"   {param}: {value}")

    baseline_model = xgb.XGBRegressor(**baseline_params)
    baseline_model.fit(X_train, y_train)

    baseline_train_pred = baseline_model.predict(X_train)
    baseline_test_pred = baseline_model.predict(X_test)

    baseline_metrics = {
        'train_r2': r2_score(y_train, baseline_train_pred),
        'test_r2': r2_score(y_test, baseline_test_pred),
        'train_mae': mean_absolute_error(y_train, baseline_train_pred),
        'test_mae': mean_absolute_error(y_test, baseline_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, baseline_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, baseline_test_pred)),
    }

    print(f"\n   Test R²:   {baseline_metrics['test_r2']:.4f}")
    print(f"   Test MAE:  ${baseline_metrics['test_mae']:,.2f}")
    print(f"   Test RMSE: ${baseline_metrics['test_rmse']:,.2f}")

    # Tuned model
    print("\n2. TUNED MODEL (Optimized Parameters)")
    print("-"*60)
    for param, value in best_params.items():
        print(f"   {param}: {value}")

    tuned_params = {**best_params, 'random_state': 42, 'n_jobs': -1, 'verbosity': 0}
    tuned_model = xgb.XGBRegressor(**tuned_params)
    tuned_model.fit(X_train, y_train)

    tuned_train_pred = tuned_model.predict(X_train)
    tuned_test_pred = tuned_model.predict(X_test)

    tuned_metrics = {
        'train_r2': r2_score(y_train, tuned_train_pred),
        'test_r2': r2_score(y_test, tuned_test_pred),
        'train_mae': mean_absolute_error(y_train, tuned_train_pred),
        'test_mae': mean_absolute_error(y_test, tuned_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, tuned_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, tuned_test_pred)),
    }

    print(f"\n   Test R²:   {tuned_metrics['test_r2']:.4f}")
    print(f"   Test MAE:  ${tuned_metrics['test_mae']:,.2f}")
    print(f"   Test RMSE: ${tuned_metrics['test_rmse']:,.2f}")

    # Improvement summary
    print("\n3. IMPROVEMENT SUMMARY")
    print("-"*60)

    r2_improvement = tuned_metrics['test_r2'] - baseline_metrics['test_r2']
    r2_pct_improvement = (r2_improvement / baseline_metrics['test_r2']) * 100

    mae_improvement = baseline_metrics['test_mae'] - tuned_metrics['test_mae']
    mae_pct_improvement = (mae_improvement / baseline_metrics['test_mae']) * 100

    rmse_improvement = baseline_metrics['test_rmse'] - tuned_metrics['test_rmse']
    rmse_pct_improvement = (rmse_improvement / baseline_metrics['test_rmse']) * 100

    print(f"   R² Score:  {baseline_metrics['test_r2']:.4f} → {tuned_metrics['test_r2']:.4f} ({r2_pct_improvement:+.2f}%)")
    print(f"   MAE:       ${baseline_metrics['test_mae']:,.0f} → ${tuned_metrics['test_mae']:,.0f} ({mae_pct_improvement:+.2f}%)")
    print(f"   RMSE:      ${baseline_metrics['test_rmse']:,.0f} → ${tuned_metrics['test_rmse']:,.0f} ({rmse_pct_improvement:+.2f}%)")

    if r2_improvement > 0:
        print(f"\n   ✓ Tuning improved the model!")
        print(f"   ✓ Predictions are ${mae_improvement:,.0f} more accurate on average")
    else:
        print(f"\n   ✗ Tuning did not improve the model")
        print(f"     The baseline parameters were already well-optimized")

    return baseline_metrics, tuned_metrics, baseline_model, tuned_model


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def create_visualizations(cv_results, baseline_metrics, tuned_metrics, best_params):
    """Create and save visualizations for the tuning results."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    # 1. Baseline vs Tuned comparison
    print("\n1. Creating Baseline vs Tuned comparison plot...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = ['Baseline\nXGBoost', 'Tuned\nXGBoost']
    colors = ['steelblue', 'darkorange']
    x_pos = np.arange(len(models))

    # R² comparison
    r2_values = [baseline_metrics['test_r2'], tuned_metrics['test_r2']]
    bars1 = axes[0].bar(x_pos, r2_values, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(models)
    axes[0].set_ylim(min(r2_values) * 0.95, max(r2_values) * 1.02)
    for bar, val in zip(bars1, r2_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # MAE comparison
    mae_values = [baseline_metrics['test_mae'], tuned_metrics['test_mae']]
    bars2 = axes[1].bar(x_pos, mae_values, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('Mean Absolute Error ($)', fontsize=12)
    axes[1].set_title('MAE Comparison (lower is better)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(models)
    for bar, val in zip(bars2, mae_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                     f'${val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # RMSE comparison
    rmse_values = [baseline_metrics['test_rmse'], tuned_metrics['test_rmse']]
    bars3 = axes[2].bar(x_pos, rmse_values, color=colors, alpha=0.8, edgecolor='black')
    axes[2].set_ylabel('Root Mean Squared Error ($)', fontsize=12)
    axes[2].set_title('RMSE Comparison (lower is better)', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(models)
    for bar, val in zip(bars3, rmse_values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                     f'${val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tuning_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/tuning_comparison.png")

    # 2. Cross-validation scores distribution
    print("\n2. Creating CV scores distribution plot...")

    cv_results_df = pd.DataFrame(cv_results)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot distribution of test scores
    test_scores = cv_results_df['mean_test_score']
    ax.hist(test_scores, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(test_scores.max(), color='red', linestyle='--', linewidth=2,
               label=f'Best: {test_scores.max():.4f}')
    ax.axvline(test_scores.mean(), color='green', linestyle='--', linewidth=2,
               label=f'Mean: {test_scores.mean():.4f}')

    ax.set_xlabel('Cross-Validation R² Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of CV Scores Across Parameter Combinations', fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cv_scores_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/cv_scores_distribution.png")

    # 3. Parameter importance heatmap (correlation with score)
    print("\n3. Creating parameter importance analysis...")

    # Get parameter columns
    param_cols = [col for col in cv_results_df.columns if col.startswith('param_')]

    if len(param_cols) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate correlation of each parameter with test score
        correlations = []
        param_names = []

        for col in param_cols:
            param_name = col.replace('param_', '')
            try:
                # Convert to numeric if possible
                values = pd.to_numeric(cv_results_df[col], errors='coerce')
                if values.notna().sum() > 0:
                    corr = values.corr(cv_results_df['mean_test_score'])
                    if not np.isnan(corr):
                        correlations.append(corr)
                        param_names.append(param_name)
            except:
                continue

        if correlations:
            # Sort by absolute correlation
            sorted_idx = np.argsort(np.abs(correlations))[::-1]
            correlations = [correlations[i] for i in sorted_idx]
            param_names = [param_names[i] for i in sorted_idx]

            colors = ['green' if c > 0 else 'red' for c in correlations]
            bars = ax.barh(param_names, correlations, color=colors, alpha=0.7, edgecolor='black')

            ax.axvline(x=0, color='black', linewidth=0.8)
            ax.set_xlabel('Correlation with CV Score', fontsize=12)
            ax.set_ylabel('Parameter', fontsize=12)
            ax.set_title('Parameter Impact on Model Performance', fontsize=14, fontweight='bold')

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'parameter_importance.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   Saved: {OUTPUT_DIR}/parameter_importance.png")


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(best_params, baseline_metrics, tuned_metrics, cv_results, search_time):
    """Save tuning results to files."""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    # 1. Save best parameters as JSON
    params_path = os.path.join(OUTPUT_DIR, 'best_params.json')
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\n1. Best parameters saved to: {params_path}")

    # 2. Save comparison metrics
    comparison_df = pd.DataFrame({
        'Model': ['Baseline XGBoost', 'Tuned XGBoost'],
        'Test_R2': [baseline_metrics['test_r2'], tuned_metrics['test_r2']],
        'Test_MAE': [baseline_metrics['test_mae'], tuned_metrics['test_mae']],
        'Test_RMSE': [baseline_metrics['test_rmse'], tuned_metrics['test_rmse']],
        'Train_R2': [baseline_metrics['train_r2'], tuned_metrics['train_r2']],
        'Train_MAE': [baseline_metrics['train_mae'], tuned_metrics['train_mae']],
        'Train_RMSE': [baseline_metrics['train_rmse'], tuned_metrics['train_rmse']],
    })

    comparison_path = os.path.join(OUTPUT_DIR, 'tuning_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"2. Comparison metrics saved to: {comparison_path}")

    # 3. Save CV results
    cv_results_df = pd.DataFrame(cv_results)
    cv_path = os.path.join(OUTPUT_DIR, 'cv_results.csv')
    cv_results_df.to_csv(cv_path, index=False)
    print(f"3. CV results saved to: {cv_path}")

    # 4. Save tuning summary
    summary = {
        'search_time_seconds': search_time,
        'best_cv_score': float(cv_results_df['mean_test_score'].max()),
        'baseline_test_r2': baseline_metrics['test_r2'],
        'tuned_test_r2': tuned_metrics['test_r2'],
        'improvement_r2': tuned_metrics['test_r2'] - baseline_metrics['test_r2'],
        'improvement_mae': baseline_metrics['test_mae'] - tuned_metrics['test_mae'],
        'best_params': best_params
    }

    summary_path = os.path.join(OUTPUT_DIR, 'tuning_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"4. Tuning summary saved to: {summary_path}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

def print_final_summary(best_params, baseline_metrics, tuned_metrics, search_time):
    """Print final summary of tuning results."""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("="*60)

    r2_improvement = tuned_metrics['test_r2'] - baseline_metrics['test_r2']
    mae_improvement = baseline_metrics['test_mae'] - tuned_metrics['test_mae']

    print(f"""
    SEARCH TIME: {search_time:.1f} seconds ({search_time/60:.1f} minutes)

    BEST PARAMETERS FOUND:
    """)
    for param, value in best_params.items():
        print(f"      {param}: {value}")

    print(f"""
    PERFORMANCE COMPARISON:

                        Baseline        Tuned           Change
    R² Score:           {baseline_metrics['test_r2']:.4f}          {tuned_metrics['test_r2']:.4f}          {r2_improvement:+.4f}
    MAE:                ${baseline_metrics['test_mae']:,.0f}        ${tuned_metrics['test_mae']:,.0f}        ${-mae_improvement:,.0f}
    RMSE:               ${baseline_metrics['test_rmse']:,.0f}       ${tuned_metrics['test_rmse']:,.0f}       ${-(baseline_metrics['test_rmse'] - tuned_metrics['test_rmse']):,.0f}

    OUTPUT FILES:
    - output/best_params.json         : Best hyperparameters (JSON)
    - output/tuning_comparison.csv    : Baseline vs Tuned metrics
    - output/tuning_comparison.png    : Visual comparison
    - output/cv_scores_distribution.png : CV scores distribution
    - output/parameter_importance.png : Parameter impact analysis
    - output/cv_results.csv           : Full cross-validation results
    - output/tuning_summary.json      : Summary statistics
    """)

    print("="*60)
    print("HYPERPARAMETER TUNING COMPLETE!")
    print("="*60)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='XGBoost Hyperparameter Tuning for Housing Price Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hyperparameter_tuning.py                    # Default: RandomizedSearchCV, 50 iterations
  python hyperparameter_tuning.py --n-iter 100      # More iterations for better results
  python hyperparameter_tuning.py --method grid     # GridSearchCV (slower but thorough)
  python hyperparameter_tuning.py --cv 10           # Use 10-fold cross-validation
        """
    )

    parser.add_argument(
        '--method',
        choices=['random', 'grid'],
        default='random',
        help='Search method: random (faster) or grid (exhaustive). Default: random'
    )

    parser.add_argument(
        '--n-iter',
        type=int,
        default=50,
        help='Number of iterations for RandomizedSearchCV. Default: 50'
    )

    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Number of cross-validation folds. Default: 5'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("XGBOOST HYPERPARAMETER TUNING")
    print("King County Housing Price Prediction")
    print("="*60)

    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)

    # Run hyperparameter search
    best_params, best_score, cv_results, search_time = run_hyperparameter_search(
        X_train, y_train,
        method=args.method,
        n_iter=args.n_iter,
        cv=args.cv
    )

    # Evaluate baseline vs tuned model
    baseline_metrics, tuned_metrics, baseline_model, tuned_model = evaluate_models(
        X_train, X_test, y_train, y_test, best_params
    )

    # Create visualizations
    create_visualizations(cv_results, baseline_metrics, tuned_metrics, best_params)

    # Save results
    save_results(best_params, baseline_metrics, tuned_metrics, cv_results, search_time)

    # Print final summary
    print_final_summary(best_params, baseline_metrics, tuned_metrics, search_time)

    return best_params, tuned_metrics


if __name__ == "__main__":
    main()
