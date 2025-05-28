import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import TargetEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking graphs
plt.style.use('seaborn-v0_8')
sns.set_theme()

def prepare_features(df):
    """
    Prepare features using TargetEncoder for categorical variables and add advanced feature engineering
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
    
    # Combine all features
    feature_cols = numeric_features + cat_features
    df_model = df_model[['timestamp', 'event_date', 'price'] + feature_cols]
    
    # Remove any remaining infinite values and NaN
    df_model = df_model.replace([np.inf, -np.inf], np.nan)
    df_model = df_model.dropna()
    
    print(f"\nFinal feature count: {len(feature_cols)}")
    print("\nFeatures used in model:")
    print(feature_cols)
    
    return df_model

def build_xgboost_model(df_model):
    """
    Build and evaluate XGBoost model with time series cross-validation
    """
    print("\nBuilding XGBoost model...")
    
    X = df_model.drop(columns=['price', 'timestamp', 'event_date'])
    y = df_model['price']
    
    # Time-based cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=42  # Added for reproducibility
    )
    
    # Initialize metrics storage
    cv_scores = []
    cv_rmse = []
    cv_predictions = []
    cv_actuals = []
    
    print("\nPerforming time series cross-validation...")
    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Fit model
        model.fit(X_train, y_train,
                 eval_set=[(X_test, y_test)],
                 verbose=False)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Store predictions and actuals for later analysis
        cv_predictions.extend(y_pred)
        cv_actuals.extend(y_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        cv_scores.append(r2)
        cv_rmse.append(rmse)
        
        print(f"Fold {fold}:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: ${rmse:.2f}")
    
    print("\nCross-validation results:")
    print(f"Average R² Score: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
    print(f"Average RMSE: ${np.mean(cv_rmse):.2f} (±${np.std(cv_rmse):.2f})")
    
    # Analyze prediction errors
    cv_predictions = np.array(cv_predictions)
    cv_actuals = np.array(cv_actuals)
    errors = cv_actuals - cv_predictions
    
    print("\nError Analysis:")
    print(f"Mean Error: ${np.mean(errors):.2f}")
    print(f"Median Error: ${np.median(errors):.2f}")
    print(f"Error Standard Deviation: ${np.std(errors):.2f}")
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=50)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Error ($)')
    plt.ylabel('Count')
    plt.savefig('xgboost_error_distribution.png')
    print("\nSaved error distribution plot to 'xgboost_error_distribution.png'")
    
    # Feature importance visualization
    plt.figure(figsize=(12,8))
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    plt.barh(importance_df['Feature'].values[-20:], 
             importance_df['Importance'].values[-20:])
    plt.title("XGBoost Feature Importance (Top 20)")
    plt.xlabel("Feature Importance Score")
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png')
    print("\nSaved feature importance plot to 'xgboost_feature_importance.png'")
    
    # Print top 10 most important features
    print("\nTop 10 Most Important Features:")
    for idx, row in importance_df.iloc[-10:].iloc[::-1].iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")
    
    return model

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
    
    # Prepare features
    df_model = prepare_features(df)
    
    # Build and evaluate XGBoost model
    model = build_xgboost_model(df_model)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 