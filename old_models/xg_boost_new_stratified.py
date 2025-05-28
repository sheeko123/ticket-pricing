import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from category_encoders import TargetEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking graphs
plt.style.use('seaborn-v0_8')
sns.set_theme()

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
    
    # Create event strata based on popularity
    print("Creating event strata...")
    high_popularity_events = df_model.groupby('event_name')['price'].mean().nlargest(10).index
    df_model['event_strata'] = df_model['event_name'].apply(
        lambda x: 'High' if x in high_popularity_events else 'Low'
    )
    
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
    df_model = df_model[feature_cols + ['price', 'event_strata']]
    
    # Remove any remaining infinite values and NaN
    df_model = df_model.replace([np.inf, -np.inf], np.nan)
    df_model = df_model.dropna()
    
    print(f"\nFinal feature count: {len(feature_cols)}")
    print("\nFeatures used in model:")
    print(feature_cols)
    
    return df_model

def build_xgboost_model_stratified(df_model):
    """
    Build and evaluate XGBoost model with stratified grouped cross-validation
    """
    print("\nBuilding XGBoost model with stratified grouped cross-validation...")
    
    # Select features for the model
    exclude_cols = ['price', 'event_strata']
    feature_cols = [col for col in df_model.columns if col not in exclude_cols]
    
    X = df_model[feature_cols].astype(float)
    y = df_model['price'].astype(float)
    groups = df_model['event_name']
    strata = df_model['event_strata']
    
    # Initialize StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5)
    
    # Initialize metrics storage
    cv_scores = {
        'rmse': [],
        'mse': [],
        'r2': []
    }
    
    # Initialize feature importance storage
    feature_importance = pd.DataFrame(index=feature_cols)
    
    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0
    }
    
    # Perform stratified grouped cross-validation
    print("\nPerforming stratified grouped cross-validation...")
    for fold, (train_index, test_index) in enumerate(sgkf.split(X, strata, groups=groups), 1):
        print(f"\nFold {fold}")
        
        # Split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Get predictions
        y_pred = model.predict(dtest)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Store metrics
        cv_scores['rmse'].append(rmse)
        cv_scores['mse'].append(mse)
        cv_scores['r2'].append(r2)
        
        # Store feature importance
        importance = model.get_score(importance_type='gain')
        for feature in feature_cols:
            feature_importance.loc[feature, f'fold_{fold}'] = importance.get(feature, 0)
        
        print(f"Fold {fold} Metrics:")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MSE: ${mse:.2f}")
        print(f"R-squared: {r2:.4f}")
        
        # Print strata distribution in this fold
        train_strata = df_model.iloc[train_index]['event_strata'].value_counts()
        test_strata = df_model.iloc[test_index]['event_strata'].value_counts()
        print("\nStrata distribution in this fold:")
        print("Training set:", train_strata.to_dict())
        print("Test set:", test_strata.to_dict())
    
    # Calculate average feature importance
    feature_importance['mean_importance'] = feature_importance.mean(axis=1)
    feature_importance = feature_importance.sort_values('mean_importance', ascending=False)
    
    # Calculate and print average metrics
    print("\nCross-validation Results:")
    print(f"Average RMSE: ${np.mean(cv_scores['rmse']):.2f} ± ${np.std(cv_scores['rmse']):.2f}")
    print(f"Average MSE: ${np.mean(cv_scores['mse']):.2f} ± ${np.std(cv_scores['mse']):.2f}")
    print(f"Average R-squared: {np.mean(cv_scores['r2']):.4f} ± {np.std(cv_scores['r2']):.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x=feature_importance['mean_importance'], y=feature_importance.index)
    plt.title('Feature Importance (Average across folds)')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('graphs/feature_importance_stratified.png')
    print("Saved feature importance plot to 'graphs/feature_importance_stratified.png'")
    
    # Plot actual vs predicted for the last fold
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Actual vs Predicted Prices (XGBoost) - Last Fold')
    plt.tight_layout()
    plt.savefig('graphs/price_prediction_xgboost_stratified.png')
    print("Saved price prediction plot to 'graphs/price_prediction_xgboost_stratified.png'")
    
    return model, X_test, y_test, feature_importance

def main():
    # Load the data
    print("Loading data...")
    file_path = 'prepared_price_prediction.csv'
    df = pd.read_csv(file_path)
    
    # Analyze floor tickets
    print("\n=== FLOOR TICKETS ANALYSIS ===")
    df_floor = df[df['is_ga_floor'] == 1].copy()
    print(f"Number of floor tickets: {len(df_floor)}")
    
    # Prepare features for floor tickets
    df_model_floor = prepare_features(df_floor)
    
    # Build and evaluate XGBoost model for floor tickets
    print("\nFloor Tickets Model Performance:")
    model_floor, X_test_floor, y_test_floor, feature_importance_floor = build_xgboost_model_stratified(df_model_floor)
    
    # Analyze non-floor tickets
    print("\n=== NON-FLOOR TICKETS ANALYSIS ===")
    df_non_floor = df[df['is_ga_floor'] == 0].copy()
    print(f"Number of non-floor tickets: {len(df_non_floor)}")
    
    # Prepare features for non-floor tickets
    df_model_non_floor = prepare_features(df_non_floor)
    
    # Build and evaluate XGBoost model for non-floor tickets
    print("\nNon-Floor Tickets Model Performance:")
    model_non_floor, X_test_non_floor, y_test_non_floor, feature_importance_non_floor = build_xgboost_model_stratified(df_model_non_floor)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 