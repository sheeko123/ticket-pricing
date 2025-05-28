"""
XGBoost model implementation for ticket price prediction.
This module provides a class for training and using XGBoost models to predict ticket prices.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GroupKFold
from sklearn.preprocessing import TargetEncoder
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TicketPriceModel:
    """
    A class for training and using XGBoost models to predict ticket prices.
    Uses different cross-validation strategies based on ticket type for optimal performance.
    """
    
    def __init__(self, model_type: str = 'auto'):
        """
        Initialize the model.
        
        Args:
            model_type: Type of model to use ('auto', 'timeseries', 'group', 'stratified')
                       'auto' will choose the best model type based on ticket type
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.metrics = {}
        self.encoder = TargetEncoder()
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for modeling.
        
        Args:
            df: Input DataFrame with raw features
            
        Returns:
            DataFrame with prepared features
        """
        logger.info("Preparing features for modeling...")
        
        # Add non-linear features
        df['days_until_event_squared'] = df['days_until_event'] ** 2
        df['quantity_squared'] = df['quantity'] ** 2
        
        # Add interaction terms
        df['weekend_x_last_week'] = df['is_weekend_event'] * df['last_48_hours']
        df['days_x_quantity'] = df['days_until_event'] * df['quantity']
        
        # Calculate venue and artist metrics
        venue_stats = df.groupby('venue')['price'].agg(['mean', 'std']).reset_index()
        venue_stats.columns = ['venue', 'venue_mean_price', 'venue_price_std']
        df = df.merge(venue_stats, on='venue', how='left')
        
        artist_stats = df.groupby('event_name')['price'].agg(['mean', 'std']).reset_index()
        artist_stats.columns = ['event_name', 'artist_mean_price', 'artist_price_std']
        df = df.merge(artist_stats, on='event_name', how='left')
        
        # Add temporal encoding
        df['event_month'] = pd.to_datetime(df['event_date']).dt.month
        df['event_year'] = pd.to_datetime(df['event_date']).dt.year
        
        # Add price normalization features
        df['price_ratio'] = df['price'] / df['venue_mean_price']
        df['price_zscore'] = (df['price'] - df['venue_mean_price']) / df['venue_price_std']
        
        # Add demand-supply features
        df['listings_per_section'] = df.groupby(['venue', 'section'])['price'].transform('count')
        df['venue_popularity'] = df.groupby('venue')['price'].transform('count')
        df['artist_demand'] = df.groupby('event_name')['price'].transform('count')
        
        # Encode categorical features
        categorical_cols = ['zone', 'section', 'row', 'venue', 'event_name', 'Category']
        df[categorical_cols] = self.encoder.fit_transform(df[categorical_cols], df['price'])
        
        # Select final features
        feature_cols = [
            'quantity', 'days_until_event', 'is_weekend_event', 'event_month', 'event_year',
            'last_48_hours', 'premium_section', 'bulk_discount', 'weekend_x_last_week',
            'demand_decay', 'price_ratio', 'listings_per_section', 'venue_popularity',
            'artist_demand', 'is_ga_floor', 'zone', 'section', 'row', 'venue', 'event_name',
            'Category', 'standardized_zone'
        ]
        
        logger.info(f"Final feature count: {len(feature_cols)}")
        logger.info(f"\nFeatures used in model:\n{feature_cols}")
        
        return df[feature_cols]
    
    def train(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None) -> Dict:
        """
        Train the model using the appropriate cross-validation strategy.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            groups: Optional group labels for grouped cross-validation
            
        Returns:
            Dictionary containing model metrics
        """
        logger.info(f"\nBuilding XGBoost model with {self.model_type} cross-validation...")
        
        # Determine best model type based on ticket type if auto
        if self.model_type == 'auto':
            is_floor = X['is_ga_floor'].mean() > 0.5
            self.model_type = 'timeseries' if is_floor else 'stratified'
            logger.info(f"Auto-selected {self.model_type} model for {'floor' if is_floor else 'non-floor'} tickets")
        
        # Initialize cross-validation
        if self.model_type == 'timeseries':
            cv = TimeSeriesSplit(n_splits=5)
            cv_splits = cv.split(X)
        elif self.model_type == 'group':
            cv = GroupKFold(n_splits=5)
            cv_splits = cv.split(X, y, groups)
        else:  # stratified
            cv = GroupKFold(n_splits=5)
            cv_splits = cv.split(X, y, groups)
        
        # Initialize metrics storage
        rmse_scores = []
        mse_scores = []
        r2_scores = []
        
        # Perform cross-validation
        logger.info(f"\nPerforming {self.model_type} cross-validation...")
        
        for fold, (train_idx, test_idx) in enumerate(cv_splits, 1):
            logger.info(f"\nFold {fold}")
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            mse = np.mean((y_test - y_pred) ** 2)
            r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
            
            rmse_scores.append(rmse)
            mse_scores.append(mse)
            r2_scores.append(r2)
            
            logger.info(f"Fold {fold} Metrics:")
            logger.info(f"RMSE: ${rmse:.2f}")
            logger.info(f"MSE: ${mse:.2f}")
            logger.info(f"R-squared: {r2:.4f}")
            
            # Store feature importance from the last fold
            if fold == 5:
                self.model = model
                self.feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
        
        # Calculate average metrics
        self.metrics = {
            'rmse': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'mse': np.mean(mse_scores),
            'mse_std': np.std(mse_scores),
            'r2': np.mean(r2_scores),
            'r2_std': np.std(r2_scores)
        }
        
        logger.info("\nCross-validation Results:")
        logger.info(f"Average RMSE: ${self.metrics['rmse']:.2f} ± ${self.metrics['rmse_std']:.2f}")
        logger.info(f"Average MSE: ${self.metrics['mse']:.2f} ± ${self.metrics['mse_std']:.2f}")
        logger.info(f"Average R-squared: {self.metrics['r2']:.4f} ± {self.metrics['r2_std']:.4f}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with feature importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Model has not been trained yet")
        
        return self.feature_importance
    
    def get_metrics(self) -> Dict:
        """
        Get model performance metrics.
        
        Returns:
            Dictionary containing model metrics
        """
        if not self.metrics:
            raise ValueError("Model has not been trained yet")
        
        return self.metrics 