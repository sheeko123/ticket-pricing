import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import TargetEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import shap
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
    
    return df_model, feature_cols

def build_xgboost_model(df_model, feature_cols, ticket_type):
    """
    Build and evaluate XGBoost model with time series cross-validation
    """
    print(f"\nBuilding XGBoost model for {ticket_type}...")
    
    X = df_model.drop(columns=['price'])
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
        random_state=42
    )
    
    # Initialize metrics storage
    cv_scores = []
    cv_rmse = []
    cv_predictions = []
    cv_actuals = []
    X_test_final = None
    y_test_final = None
    
    print("\nPerforming time series cross-validation...")
    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Store the last fold's test data for SHAP analysis
        if fold == 5:
            X_test_final = X_test
            y_test_final = y_test
        
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
    
    # Print top 10 most important features
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    for idx, row in importance_df.iloc[:10].iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")
    
    # SHAP Analysis
    print("\nPerforming SHAP analysis...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_final)
    
    # Create SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_final, feature_names=feature_cols, show=False)
    plt.title(f'SHAP Summary Plot - {ticket_type}')
    plt.tight_layout()
    plt.savefig(f'shap_summary_{ticket_type.lower().replace(" ", "_")}.png')
    print(f"Saved SHAP summary plot to 'shap_summary_{ticket_type.lower().replace(' ', '_')}.png'")
    
    # Create SHAP dependence plots for top 3 features
    top_features = importance_df['Feature'].iloc[:3].tolist()
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, X_test_final, feature_names=feature_cols, show=False)
        plt.title(f'SHAP Dependence Plot - {feature} ({ticket_type})')
        plt.tight_layout()
        plt.savefig(f'shap_dependence_{feature}_{ticket_type.lower().replace(" ", "_")}.png')
        print(f"Saved SHAP dependence plot for {feature} to 'shap_dependence_{feature}_{ticket_type.lower().replace(' ', '_')}.png'")
    
    return model

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
    df_model_floor, feature_cols = prepare_features(df_floor)
    
    # Build and evaluate XGBoost model for floor tickets
    print("\nFloor Tickets Model Performance:")
    model_floor = build_xgboost_model(df_model_floor, feature_cols, "Floor Tickets")
    
    # Analyze non-floor tickets
    print("\n=== NON-FLOOR TICKETS ANALYSIS ===")
    df_non_floor = df[df['is_ga_floor'] == 0].copy()
    print(f"Number of non-floor tickets: {len(df_non_floor)}")
    
    # Prepare features for non-floor tickets
    df_model_non_floor, feature_cols = prepare_features(df_non_floor)
    
    # Build and evaluate XGBoost model for non-floor tickets
    print("\nNon-Floor Tickets Model Performance:")
    model_non_floor = build_xgboost_model(df_model_non_floor, feature_cols, "Non-Floor Tickets")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 