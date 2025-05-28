import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
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
    
    # 2. Price boxplot by standardized zone
    sns.boxplot(data=df, x='standardized_zone', y='price', ax=ax2)
    ax2.set_title('Price Distribution by Zone')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    # 3. Price vs Days Until Event scatter plot with regression line
    sns.regplot(data=df, x='days_until_event', y='price', scatter_kws={'alpha':0.3}, ax=ax3)
    ax3.set_title('Price vs Days Until Event with Trend')
    
    # 4. Average price by event (top 10 events)
    top_events = df.groupby('event_name')['price'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=top_events.index, y=top_events.values, ax=ax4)
    ax4.set_title('Average Price by Event (Top 10)')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('price_analysis_tscv.png')
    print("Saved price analysis visualizations to 'price_analysis_tscv.png'")
    
    # Print summary statistics
    print("\nPrice Statistics:")
    print(df['price'].describe())

def prepare_features(df):
    """
    Prepare features for modeling
    """
    print("\nPreparing features for modeling...")
    
    # Create a copy of the dataframe
    df_model = df.copy()
    
    # Convert timestamp and event_date to datetime
    df_model['timestamp'] = pd.to_datetime(df_model['timestamp'])
    df_model['event_date'] = pd.to_datetime(df_model['event_date'])
    
    # Sort by timestamp for time series analysis
    df_model = df_model.sort_values('timestamp')
    
    # 1. Non-Linear Feature Engineering
    print("Adding non-linear features...")
    df_model['last_48_hours'] = (df_model['days_until_event'] <= 2).astype(int)
    df_model['premium_section'] = (df_model['section'].str.contains('VIP|Front|Lounge', na=False)).astype(int)
    df_model['bulk_discount'] = np.where(df_model['quantity'] > 4, 1, 0)
    
    # 2. Interaction Terms
    print("Adding interaction terms...")
    df_model['weekend_x_last_week'] = df_model['is_weekend_event'] * (df_model['days_until_event'] <= 7)
    
    # Calculate venue popularity and artist demand using expanding windows
    print("Calculating venue and artist metrics...")
    df_model['venue_popularity'] = df_model.groupby('venue')['price'].expanding().mean().reset_index(level=0, drop=True)
    df_model['artist_demand'] = df_model.groupby('event_name')['price'].expanding().std().reset_index(level=0, drop=True)
    
    # 3. Advanced Temporal Encoding
    print("Adding temporal encoding...")
    df_model['demand_decay'] = 1 / (1 + np.exp(-0.1 * (7 - df_model['days_until_event'])))
    
    # 4. Price Normalization
    print("Adding price normalization features...")
    df_model['median_price'] = df_model.groupby(['event_name', 'section'])['price'].expanding().median().reset_index(level=[0,1], drop=True)
    df_model['price_ratio'] = df_model['price'] / df_model['median_price']
    
    # 5. Demand-Supply Ratio
    print("Adding demand-supply features...")
    df_model['listings_per_section'] = df_model.groupby(['event_date', 'section'])['price'].expanding().count().reset_index(level=[0,1], drop=True)
    
    # List of categorical features
    cat_features = ['zone', 'section', 'row', 'venue', 'event_name', 'Category', 'standardized_zone']
    
    # Initialize Target Encoder
    encoder = TargetEncoder(cols=cat_features)
    
    # Fit-transform on entire dataset
    print("Encoding categorical features using TargetEncoder...")
    df_model[cat_features] = encoder.fit_transform(df_model[cat_features], df_model['price'])
    
    # Keep numeric features including new engineered features
    numeric_features = ['quantity', 'days_until_event', 
                       'is_weekend_event', 'event_month', 'event_year',
                       'last_48_hours', 'premium_section', 'bulk_discount',
                       'weekend_x_last_week', 'demand_decay', 'price_ratio',
                       'listings_per_section', 'venue_popularity', 'artist_demand',
                       'is_ga_floor']
    
    # Combine features
    feature_cols = numeric_features + cat_features
    df_model = df_model[feature_cols + ['price']]
    
    # Remove any remaining infinite values and NaN
    df_model = df_model.replace([np.inf, -np.inf], np.nan)
    df_model = df_model.dropna()
    
    print(f"\nFinal feature count: {len(feature_cols)}")
    print("\nFeatures used in model:")
    print(feature_cols)
    
    return df_model

def build_glm_model_tscv(df_model, model_type='tweedie'):
    """
    Build and evaluate GLM model with time series cross-validation
    """
    print(f"\nBuilding GLM model with {model_type} distribution using time series cross-validation...")
    
    # Select features for the model
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
    
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Initialize metrics storage
    cv_scores = {
        'rmse': [],
        'mse': [],
        'r2': []
    }
    
    # Perform time series cross-validation
    print("\nPerforming time series cross-validation...")
    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
        print(f"\nFold {fold}")
        
        # Split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Add constant for statsmodels
        X_train_sm = sm.add_constant(X_train)
        X_test_sm = sm.add_constant(X_test)
        
        # Fit GLM with specified distribution
        log_link = sm.families.links.Log()
        
        if model_type == 'gamma':
            model = sm.GLM(y_train, X_train_sm, family=sm.families.Gamma(link=log_link))
        else:  # tweedie
            model = sm.GLM(y_train, X_train_sm, family=sm.families.Tweedie(var_power=1.5, link=log_link))
        
        results = model.fit()
        
        # Calculate predictions
        y_pred = results.predict(X_test_sm)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Store metrics
        cv_scores['rmse'].append(rmse)
        cv_scores['mse'].append(mse)
        cv_scores['r2'].append(r2)
        
        print(f"Fold {fold} Metrics:")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MSE: ${mse:.2f}")
        print(f"R-squared: {r2:.4f}")
    
    # Calculate and print average metrics
    print("\nCross-validation Results:")
    print(f"Average RMSE: ${np.mean(cv_scores['rmse']):.2f} ± ${np.std(cv_scores['rmse']):.2f}")
    print(f"Average MSE: ${np.mean(cv_scores['mse']):.2f} ± ${np.std(cv_scores['mse']):.2f}")
    print(f"Average R-squared: {np.mean(cv_scores['r2']):.4f} ± {np.std(cv_scores['r2']):.4f}")
    
    # Plot actual vs predicted for the last fold
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title(f'Actual vs Predicted Prices ({model_type.capitalize()} GLM) - Last Fold')
    plt.tight_layout()
    plt.savefig(f'price_prediction_{model_type}_tscv.png')
    print(f"Saved price prediction plot to 'price_prediction_{model_type}_tscv.png'")
    
    return results, X_test, y_test

def main():
    # Load the data
    print("Loading data...")
    file_path = 'prepared_price_prediction.csv'
    df = pd.read_csv(file_path)
    
    # Analyze floor tickets
    print("\n=== FLOOR TICKETS ANALYSIS ===")
    df_floor = df[df['is_ga_floor'] == 1].copy()
    print(f"Number of floor tickets: {len(df_floor)}")
    
    # Analyze price distribution for floor tickets
    print("\nFloor Tickets Price Statistics:")
    print(df_floor['price'].describe())
    
    # Prepare features for floor tickets
    df_model_floor = prepare_features(df_floor)
    
    # Build and evaluate both models for floor tickets
    print("\nFloor Tickets Model Performance:")
    gamma_results_floor, X_test_floor, y_test_floor = build_glm_model_tscv(df_model_floor, model_type='gamma')
    tweedie_results_floor, _, _ = build_glm_model_tscv(df_model_floor, model_type='tweedie')
    
    # Analyze non-floor tickets
    print("\n=== NON-FLOOR TICKETS ANALYSIS ===")
    df_non_floor = df[df['is_ga_floor'] == 0].copy()
    print(f"Number of non-floor tickets: {len(df_non_floor)}")
    
    # Analyze price distribution for non-floor tickets
    print("\nNon-Floor Tickets Price Statistics:")
    print(df_non_floor['price'].describe())
    
    # Prepare features for non-floor tickets
    df_model_non_floor = prepare_features(df_non_floor)
    
    # Build and evaluate both models for non-floor tickets
    print("\nNon-Floor Tickets Model Performance:")
    gamma_results_non_floor, X_test_non_floor, y_test_non_floor = build_glm_model_tscv(df_model_non_floor, model_type='gamma')
    tweedie_results_non_floor, _, _ = build_glm_model_tscv(df_model_non_floor, model_type='tweedie')
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 