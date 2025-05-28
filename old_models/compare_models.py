import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from xg_boost_new_stratified import prepare_features as prepare_features_stratified, build_xgboost_model_stratified
from xg_boost_new_tsCV import prepare_features as prepare_features_tscv, build_xgboost_model_tscv
from xg_boost_new_groupCV import prepare_features as prepare_features_groupcv, build_xgboost_model_groupcv

def run_comparison():
    print("Loading data...")
    file_path = 'prepared_price_prediction.csv'
    df = pd.read_csv(file_path)
    
    # Initialize results storage
    results = {
        'Floor Tickets': {
            'Stratified': {'rmse': [], 'r2': []},
            'TimeSeries': {'rmse': [], 'r2': []},
            'Group': {'rmse': [], 'r2': []}
        },
        'Non-Floor Tickets': {
            'Stratified': {'rmse': [], 'r2': []},
            'TimeSeries': {'rmse': [], 'r2': []},
            'Group': {'rmse': [], 'r2': []}
        }
    }
    
    # Process floor tickets
    print("\n=== FLOOR TICKETS ANALYSIS ===")
    df_floor = df[df['is_ga_floor'] == 1].copy()
    print(f"Number of floor tickets: {len(df_floor)}")
    
    # Stratified
    print("\nRunning Stratified Model...")
    df_model_floor = prepare_features_stratified(df_floor)
    model_floor, X_test_floor, y_test_floor, feature_importance_floor = build_xgboost_model_stratified(df_model_floor)
    y_pred_floor = model_floor.predict(xgb.DMatrix(X_test_floor))
    rmse = np.sqrt(mean_squared_error(y_test_floor, y_pred_floor))
    r2 = r2_score(y_test_floor, y_pred_floor)
    results['Floor Tickets']['Stratified']['rmse'].append(rmse)
    results['Floor Tickets']['Stratified']['r2'].append(r2)
    
    # TimeSeries
    print("\nRunning TimeSeries Model...")
    df_model_floor = prepare_features_tscv(df_floor)
    model_floor, X_test_floor, y_test_floor, feature_importance_floor = build_xgboost_model_tscv(df_model_floor)
    y_pred_floor = model_floor.predict(xgb.DMatrix(X_test_floor))
    rmse = np.sqrt(mean_squared_error(y_test_floor, y_pred_floor))
    r2 = r2_score(y_test_floor, y_pred_floor)
    results['Floor Tickets']['TimeSeries']['rmse'].append(rmse)
    results['Floor Tickets']['TimeSeries']['r2'].append(r2)
    
    # Group
    print("\nRunning Group Model...")
    df_model_floor = prepare_features_groupcv(df_floor)
    model_floor, X_test_floor, y_test_floor, feature_importance_floor = build_xgboost_model_groupcv(df_model_floor)
    y_pred_floor = model_floor.predict(xgb.DMatrix(X_test_floor))
    rmse = np.sqrt(mean_squared_error(y_test_floor, y_pred_floor))
    r2 = r2_score(y_test_floor, y_pred_floor)
    results['Floor Tickets']['Group']['rmse'].append(rmse)
    results['Floor Tickets']['Group']['r2'].append(r2)
    
    # Process non-floor tickets
    print("\n=== NON-FLOOR TICKETS ANALYSIS ===")
    df_non_floor = df[df['is_ga_floor'] == 0].copy()
    print(f"Number of non-floor tickets: {len(df_non_floor)}")
    
    # Stratified
    print("\nRunning Stratified Model...")
    df_model_non_floor = prepare_features_stratified(df_non_floor)
    model_non_floor, X_test_non_floor, y_test_non_floor, feature_importance_non_floor = build_xgboost_model_stratified(df_model_non_floor)
    y_pred_non_floor = model_non_floor.predict(xgb.DMatrix(X_test_non_floor))
    rmse = np.sqrt(mean_squared_error(y_test_non_floor, y_pred_non_floor))
    r2 = r2_score(y_test_non_floor, y_pred_non_floor)
    results['Non-Floor Tickets']['Stratified']['rmse'].append(rmse)
    results['Non-Floor Tickets']['Stratified']['r2'].append(r2)
    
    # TimeSeries
    print("\nRunning TimeSeries Model...")
    df_model_non_floor = prepare_features_tscv(df_non_floor)
    model_non_floor, X_test_non_floor, y_test_non_floor, feature_importance_non_floor = build_xgboost_model_tscv(df_model_non_floor)
    y_pred_non_floor = model_non_floor.predict(xgb.DMatrix(X_test_non_floor))
    rmse = np.sqrt(mean_squared_error(y_test_non_floor, y_pred_non_floor))
    r2 = r2_score(y_test_non_floor, y_pred_non_floor)
    results['Non-Floor Tickets']['TimeSeries']['rmse'].append(rmse)
    results['Non-Floor Tickets']['TimeSeries']['r2'].append(r2)
    
    # Group
    print("\nRunning Group Model...")
    df_model_non_floor = prepare_features_groupcv(df_non_floor)
    model_non_floor, X_test_non_floor, y_test_non_floor, feature_importance_non_floor = build_xgboost_model_groupcv(df_model_non_floor)
    y_pred_non_floor = model_non_floor.predict(xgb.DMatrix(X_test_non_floor))
    rmse = np.sqrt(mean_squared_error(y_test_non_floor, y_pred_non_floor))
    r2 = r2_score(y_test_non_floor, y_pred_non_floor)
    results['Non-Floor Tickets']['Group']['rmse'].append(rmse)
    results['Non-Floor Tickets']['Group']['r2'].append(r2)
    
    # Print comparison results
    print("\n=== MODEL COMPARISON RESULTS ===")
    for ticket_type in results:
        print(f"\n{ticket_type}:")
        print("-" * 50)
        for model_type in results[ticket_type]:
            rmse = np.mean(results[ticket_type][model_type]['rmse'])
            r2 = np.mean(results[ticket_type][model_type]['r2'])
            print(f"{model_type} Model:")
            print(f"RMSE: ${rmse:.2f}")
            print(f"R-squared: {r2:.4f}")
            print()

if __name__ == "__main__":
    run_comparison() 