import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, TimeSeriesSplit
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from category_encoders import TargetEncoder
from typing import Tuple, Dict


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for modeling (shared for both models)
    """
    df_model = df.copy()
    df_model['timestamp'] = pd.to_datetime(df_model['timestamp'])
    df_model['event_date'] = pd.to_datetime(df_model['event_date'])
    df_model = df_model.sort_values('timestamp')
    df_model['last_48_hours'] = (df_model['days_until_event'] <= 2).astype(int)
    df_model['premium_section'] = (df_model['section'].str.contains('VIP|Front|Lounge', na=False)).astype(int)
    df_model['bulk_discount'] = np.where(df_model['quantity'] > 4, 1, 0)
    df_model['weekend_x_last_week'] = df_model['is_weekend_event'] * (df_model['days_until_event'] <= 7)
    df_model['venue_popularity'] = df_model.groupby('venue')['price'].expanding().mean().reset_index(level=0, drop=True)
    df_model['artist_demand'] = df_model.groupby('event_name')['price'].expanding().std().reset_index(level=0, drop=True)
    df_model['demand_decay'] = 1 / (1 + np.exp(-0.1 * (7 - df_model['days_until_event'])))
    df_model['median_price'] = df_model.groupby(['event_name', 'section'])['price'].expanding().median().reset_index(level=[0,1], drop=True)
    df_model['price_ratio'] = df_model['price'] / df_model['median_price']
    df_model['listings_per_section'] = df_model.groupby(['event_date', 'section'])['price'].expanding().count().reset_index(level=[0,1], drop=True)
    cat_features = ['zone', 'section', 'row', 'venue', 'event_name', 'Category', 'standardized_zone']
    encoder = TargetEncoder(cols=cat_features)
    df_model[cat_features] = encoder.fit_transform(df_model[cat_features], df_model['price'])
    numeric_features = ['quantity', 'days_until_event', 'is_weekend_event', 'event_month', 'event_year',
                       'last_48_hours', 'premium_section', 'bulk_discount', 'weekend_x_last_week',
                       'demand_decay', 'price_ratio', 'listings_per_section', 'venue_popularity', 'artist_demand',
                       'is_ga_floor']
    feature_cols = numeric_features + cat_features
    # For stratified model, add event_strata
    if 'event_strata' in df_model.columns:
        df_model = df_model[feature_cols + ['price', 'event_strata']]
    else:
        df_model = df_model[feature_cols + ['price']]
    df_model = df_model.replace([np.inf, -np.inf], np.nan)
    df_model = df_model.dropna()
    return df_model


def build_xgboost_model_stratified(df_model: pd.DataFrame) -> Tuple[xgb.Booster, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build and evaluate XGBoost model with stratified grouped cross-validation (for non-floor tickets)
    Returns: model, X_test, y_test, feature_importance
    """
    exclude_cols = ['price', 'event_strata']
    feature_cols = [col for col in df_model.columns if col not in exclude_cols]
    X = df_model[feature_cols].astype(float)
    y = df_model['price'].astype(float)
    groups = df_model['event_name']
    strata = df_model['event_strata']
    sgkf = StratifiedGroupKFold(n_splits=5)
    feature_importance = pd.DataFrame(index=feature_cols)
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
    # Use the last fold for test data
    for fold, (train_index, test_index) in enumerate(sgkf.split(X, strata, groups=groups), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        y_pred = model.predict(dtest)
        importance = model.get_score(importance_type='gain')
        for feature in feature_cols:
            feature_importance.loc[feature, f'fold_{fold}'] = importance.get(feature, 0)
        # Return the last fold's test data/model
        if fold == 5:
            feature_importance['mean_importance'] = feature_importance.mean(axis=1)
            feature_importance = feature_importance.sort_values('mean_importance', ascending=False)
            return model, X_test, y_test, feature_importance


def build_xgboost_model_tscv(df_model: pd.DataFrame) -> Tuple[xgb.Booster, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build and evaluate XGBoost model with time series cross-validation (for floor tickets)
    Returns: model, X_test, y_test, feature_importance
    """
    exclude_cols = ['price']
    feature_cols = [col for col in df_model.columns if col not in exclude_cols]
    X = df_model[feature_cols].astype(float)
    y = df_model['price'].astype(float)
    tscv = TimeSeriesSplit(n_splits=5)
    feature_importance = pd.DataFrame(index=feature_cols)
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
    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        y_pred = model.predict(dtest)
        importance = model.get_score(importance_type='gain')
        for feature in feature_cols:
            feature_importance.loc[feature, f'fold_{fold}'] = importance.get(feature, 0)
        if fold == 5:
            feature_importance['mean_importance'] = feature_importance.mean(axis=1)
            feature_importance = feature_importance.sort_values('mean_importance', ascending=False)
            return model, X_test, y_test, feature_importance 