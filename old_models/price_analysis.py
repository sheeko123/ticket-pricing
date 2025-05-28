import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from category_encoders import TargetEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking graphs
plt.style.use('seaborn-v0_8')
sns.set_theme()

def load_and_prepare_data(file_path):
    """
    Load and prepare the cleaned data for analysis
    """
    print("Loading cleaned data...")
    df = pd.read_csv(file_path)
    
    print("\nData Overview:")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print("\nColumns in dataset:")
    print(df.columns.tolist())
    
    return df

def analyze_price_distribution(df):
    """
    Analyze and visualize price distribution with enhanced diagnostics
    """
    print("\nAnalyzing price distribution...")
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Price distribution histogram with KDE
    sns.histplot(data=df, x='price', bins=50, kde=True, ax=ax1)
    ax1.set_title('Price Distribution with KDE')
    ax1.set_xlabel('Price ($)')
    
    # 2. Price boxplot by category
    sns.boxplot(data=df, x='Category', y='price', ax=ax2)
    ax2.set_title('Price Distribution by Category')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    # 3. Price vs Days Until Event scatter plot with regression line
    sns.regplot(data=df, x='days_until_event', y='price', scatter_kws={'alpha':0.3}, ax=ax3)
    ax3.set_title('Price vs Days Until Event with Trend')
    
    # 4. Average price by zone (top 10 zones)
    top_zones = df.groupby('zone')['price'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=top_zones.index, y=top_zones.values, ax=ax4)
    ax4.set_title('Average Price by Zone (Top 10)')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('price_analysis.png')
    print("Saved price analysis visualizations to 'price_analysis.png'")
    
    # Print summary statistics
    print("\nPrice Statistics:")
    print(df['price'].describe())

def prepare_features(df):
    """
    Prepare features using TargetEncoder for categorical variables and add advanced feature engineering
    """
    print("\nPreparing features for modeling...")
    
    # Create a copy of the dataframe and sort by timestamp
    df_model = df.copy()
    df_model = df_model.sort_values('timestamp')
    
    # 1. Non-Linear Feature Engineering
    print("Adding non-linear features...")
    df_model['last_48_hours'] = (df_model['days_until_event'] <= 2).astype(int)
    df_model['premium_section'] = (df_model['section'].str.contains('VIP|Front|Lounge', na=False)).astype(int)
    df_model['bulk_discount'] = np.where(df_model['quantity'] > 4, 1, 0)
    
    # 2. Interaction Terms (without data leakage)
    print("Adding interaction terms...")
    df_model['weekend_x_last_week'] = df_model['is_weekend_event'] * (df_model['days_until_event'] <= 7)
    
    # Calculate venue popularity and artist demand using expanding windows
    print("Calculating venue and artist metrics...")
    df_model['venue_popularity'] = df_model.groupby('venue')['price'].expanding().mean().reset_index(level=0, drop=True)
    df_model['artist_demand'] = df_model.groupby('event_name')['price'].expanding().std().reset_index(level=0, drop=True)
    
    # 3. Advanced Temporal Encoding
    print("Adding temporal encoding...")
    df_model['demand_decay'] = 1 / (1 + np.exp(-0.1 * (7 - df_model['days_until_event'])))
    
    # 4. Price Normalization (using expanding window to prevent data leakage)
    print("Adding price normalization features...")
    df_model['median_price'] = df_model.groupby(['event_name', 'section'])['price'].expanding().median().reset_index(level=[0,1], drop=True)
    df_model['price_ratio'] = df_model['price'] / df_model['median_price']
    
    # 5. Demand-Supply Ratio (using expanding window)
    print("Adding demand-supply features...")
    df_model['listings_per_section'] = df_model.groupby(['event_date', 'section'])['price'].expanding().count().reset_index(level=[0,1], drop=True)
    
    # List of high-value categorical features
    cat_features = ['zone', 'section', 'row', 'venue', 'event_name', 'Category']
    
    # Initialize Target Encoder
    encoder = TargetEncoder(cols=cat_features)
    
    # Fit-transform on entire dataset
    print("Encoding categorical features using TargetEncoder...")
    df_model[cat_features] = encoder.fit_transform(df_model[cat_features], df_model['price'])
    
    # Keep other numeric features including new engineered features
    numeric_features = ['quantity', 'days_until_event', 
                       'is_weekend_event', 'event_month', 'event_year',
                       'last_48_hours', 'premium_section', 'bulk_discount',
                       'weekend_x_last_week', 'demand_decay', 'price_ratio',
                       'listings_per_section', 'venue_popularity', 'artist_demand']
    
    # Combine features
    df_model = df_model[numeric_features + cat_features + ['price']]
    
    # Remove any remaining infinite values and NaN
    df_model = df_model.replace([np.inf, -np.inf], np.nan)
    df_model = df_model.dropna()
    
    print(f"\nFinal feature count: {len(df_model.columns)}")
    print("\nFeatures used in model:")
    print(df_model.columns.tolist())
    
    return df_model

def calculate_vif(X):
    """
    Calculate Variance Inflation Factor for features
    """
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

def analyze_errors(y_true, y_pred, df):
    """
    Analyze prediction errors to identify patterns in over/under predictions
    """
    errors = y_true - y_pred
    df['error'] = errors
    
    print("\nWorst Overpredictions (Predicted > Actual):")
    print(df[df['error'] < -100].sort_values('error').head())
    print("\nWorst Underpredictions (Predicted < Actual):")
    print(df[df['error'] > 100].sort_values('error', ascending=False).head())
    
    # Additional error analysis
    print("\nError Statistics:")
    print(f"Mean Error: ${errors.mean():.2f}")
    print(f"Median Error: ${errors.median():.2f}")
    print(f"Error Standard Deviation: ${errors.std():.2f}")
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=50)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Error ($)')
    plt.ylabel('Count')
    plt.savefig('error_distribution.png')
    print("\nSaved error distribution plot to 'error_distribution.png'")

def compare_models(gamma_results, tweedie_results, X_test, y_test):
    """
    Compare performance of Gamma and Tweedie GLM models
    """
    print("\n=== Model Comparison ===")
    
    # Get predictions
    X_test_sm = sm.add_constant(X_test)
    gamma_pred = gamma_results.predict(X_test_sm)
    tweedie_pred = tweedie_results.predict(X_test_sm)
    
    # Calculate metrics for both models
    metrics = {
        'Gamma GLM': {
            'RMSE': np.sqrt(mean_squared_error(y_test, gamma_pred)),
            'MSE': mean_squared_error(y_test, gamma_pred),
            'R2': r2_score(y_test, gamma_pred),
            'AIC': gamma_results.aic,
            'BIC': gamma_results.bic
        },
        'Tweedie GLM': {
            'RMSE': np.sqrt(mean_squared_error(y_test, tweedie_pred)),
            'MSE': mean_squared_error(y_test, tweedie_pred),
            'R2': r2_score(y_test, tweedie_pred),
            'AIC': tweedie_results.aic,
            'BIC': tweedie_results.bic
        }
    }
    
    # Print comparison table
    print("\nModel Performance Comparison:")
    print("-" * 50)
    print(f"{'Metric':<15} {'Gamma GLM':<15} {'Tweedie GLM':<15} {'Difference':<15}")
    print("-" * 50)
    
    for metric in ['RMSE', 'MSE', 'R2', 'AIC', 'BIC']:
        gamma_val = metrics['Gamma GLM'][metric]
        tweedie_val = metrics['Tweedie GLM'][metric]
        diff = tweedie_val - gamma_val
        print(f"{metric:<15} ${gamma_val:<14.2f} ${tweedie_val:<14.2f} ${diff:<14.2f}")
    
    # Plot comparison of predictions
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, gamma_pred, alpha=0.5, label='Gamma GLM')
    plt.scatter(y_test, tweedie_pred, alpha=0.5, label='Tweedie GLM')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Model Comparison: Actual vs Predicted Prices')
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("\nSaved model comparison plot to 'model_comparison.png'")
    
    # Compare error distributions
    gamma_errors = y_test - gamma_pred
    tweedie_errors = y_test - tweedie_pred
    
    plt.figure(figsize=(12, 6))
    sns.kdeplot(gamma_errors, label='Gamma GLM')
    sns.kdeplot(tweedie_errors, label='Tweedie GLM')
    plt.xlabel('Prediction Error ($)')
    plt.ylabel('Density')
    plt.title('Error Distribution Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig('error_comparison.png')
    print("Saved error distribution comparison to 'error_comparison.png'")

def build_glm_model(df_model, model_type='tweedie'):
    """
    Build and evaluate GLM model with specified distribution
    """
    print(f"\nBuilding GLM model with {model_type} distribution...")
    
    # Select features for the model (exclude non-feature columns)
    exclude_cols = ['price']
    feature_cols = [col for col in df_model.columns if col not in exclude_cols]
    
    X = df_model[feature_cols].astype(float)
    y = df_model['price'].astype(float)
    
    # Calculate VIF for all features
    print("\nCalculating VIF for features...")
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data = vif_data.sort_values('VIF', ascending=False)
    print("\nTop 10 highest VIF values:")
    print(vif_data.head(10))
    
    # Remove features with high VIF
    high_vif_features = vif_data[vif_data['VIF'] > 10]['Variable'].tolist()
    if high_vif_features:
        print(f"\nRemoving features with high VIF: {high_vif_features}")
        X = X.drop(columns=high_vif_features)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Add constant for statsmodels
    X_train_sm = sm.add_constant(X_train)
    
    # Fit GLM with specified distribution
    print(f"\nFitting GLM model with {model_type} distribution...")
    log_link = sm.families.links.Log()
    
    if model_type == 'gamma':
        model = sm.GLM(y_train, X_train_sm, family=sm.families.Gamma(link=log_link))
    else:  # tweedie
        model = sm.GLM(y_train, X_train_sm, family=sm.families.Tweedie(var_power=1.5, link=log_link))
    
    results = model.fit()
    
    print(f"\n{model_type.capitalize()} GLM Model Summary:")
    print(results.summary())
    
    # Calculate predictions and metrics
    X_test_sm = sm.add_constant(X_test)
    y_pred = results.predict(X_test_sm)
    
    # Calculate model metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print("\nModel Performance Metrics:")
    print(f"Root Mean Squared Error: ${rmse:.2f}")
    print(f"Mean Squared Error: ${mse:.2f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title(f'Actual vs Predicted Prices ({model_type.capitalize()} GLM)')
    plt.tight_layout()
    plt.savefig(f'price_prediction_{model_type}.png')
    print(f"Saved price prediction plot to 'price_prediction_{model_type}.png'")
    
    # Analyze prediction errors
    analyze_errors(y_test, y_pred, pd.DataFrame({'price': y_test}))
    
    return results, X_test, y_test

def main():
    # Load the data
    print("Loading data...")
    file_path = 'prepared_price_prediction.csv'
    df = pd.read_csv(file_path)
    
    # Print the first line
    print("\nFirst line of the CSV:")
    print(df.iloc[0])
    
    # Print unique standardized zone values
    print("\nUnique standardized zone values:")
    print(df['standardized_zone'].unique())
    
    # Print counts and average prices for each standardized zone
    print("\nCounts and average prices for each standardized zone:")
    zone_stats = df.groupby('standardized_zone').agg(
        count=('price', 'count'),
        avg_price=('price', 'mean')
    ).sort_values('avg_price', ascending=False)
    print(zone_stats)
    
    # Print the highest and lowest priced zones
    print("\nHighest priced zone:")
    print(zone_stats.iloc[0])
    print("\nLowest priced zone:")
    print(zone_stats.iloc[-1])

if __name__ == "__main__":
    main() 