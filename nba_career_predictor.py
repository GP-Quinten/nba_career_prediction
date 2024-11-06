import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import shap
import joblib
import logging
import os
from config import PARAM_GRIDS, MINUTES_BINS, GAMES_BINS, OUTCOME

class NBACareerPredictor:
    def __init__(self):
        self.model = None
        self.features_list = None
        self.scaler = MinMaxScaler()
        self.hyperparameters = {}
        self.feature_names = None
        self.threshold = 0.5
        
    def add_features(self, df):
        """Add engineered features to the dataset"""
        logging.info("Adding engineered features...")
        
        # Create copy to avoid modifying original
        enhanced_df = df.copy()
        
        # Total features
        for col in df.columns:
            if col not in ['Name', OUTCOME, 'FG%', '3P%', 'FT%']:
                enhanced_df[f'{col}_TOT'] = df['GP'] * df[col]
        
        # Efficiency rates
        tot_cols = [col for col in enhanced_df.columns if col.endswith('_TOT')]
        for col in tot_cols:
            base_name = col.replace('_TOT', '')
            enhanced_df[f'{base_name}_RATE'] = enhanced_df[col] / np.maximum(enhanced_df['MIN_TOT'], 1)
        
        # Categorical features for minutes and games
        enhanced_df['MIN_CAT'] = pd.cut(
            enhanced_df['MIN'],
            bins=MINUTES_BINS,
            labels=['MIN<10', 'MIN<20', 'MIN<30', 'MIN>30']
        )
        enhanced_df['GP_CAT'] = pd.cut(
            enhanced_df['GP'],
            bins=GAMES_BINS,
            labels=['GP<28', 'GP<56', 'GP≥56']
        )
        
        # One-hot encode categorical features
        enhanced_df = pd.get_dummies(enhanced_df, columns=['MIN_CAT', 'GP_CAT'])
        
        logging.info(f"Added {len(enhanced_df.columns) - len(df.columns)} new features")
        return enhanced_df
    
    def preprocess_data(self, df):
        """Preprocess the dataset"""
        logging.info("Preprocessing data...")
        
        # Replace NaN values
        df = df.fillna(0)
        
        # Drop duplicates
        df = df.groupby([col for col in df.columns if col != OUTCOME])\
               .agg({OUTCOME: 'max'})\
               .reset_index()
        
        logging.info(f"Dataset contains {df['Name'].nunique()} unique players")
        
        # Store feature names
        self.features_list = [col for col in df.columns if col not in ['Name', OUTCOME]]
        
        # Split features and target
        X = df[self.features_list]
        y = df[OUTCOME]
        
        # Normalize features
        X = self.scaler.fit_transform(X)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def hyperparameter_tuning_model(self, X_train, y_train, model_type):
        """Tuning hyperparameters for the model"""
        logging.info(f"Fine-tuning {model_type} model...")
        
        # Get base model and parameter grid
        base_model = self._get_model_instance(model_type)
        param_grid = PARAM_GRIDS[model_type]
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='average_precision',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store best hyperparameters
        self.hyperparameters = grid_search.best_params_
        logging.info(f"Best parameters: {self.hyperparameters}")
        
        return grid_search.best_estimator_
    
    def train_and_test_model(self, X_train, y_train, X_test, y_test, model_type):
        """Train and evaluate model"""
        logging.info("Training final model...")
        
        # Get tuned model
        self.model = self.hyperparameter_tuning_model(X_train, y_train, model_type)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        logging.info("Model performance metrics:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.3f}")
        
        return metrics, conf_matrix, y_prob
    
    def explain_predictions(self, X):
        """Generate SHAP values for model interpretation"""
        logging.info("Generating SHAP values...")
        # if the model type is not SVM, we can use the TreeExplainer
        from sklearn.svm import SVC
        if not isinstance(self.model, SVC):
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
        else:
            # if the model type is SVM, we need to use the KernelExplainer
            explainer = shap.KernelExplainer(self.model.predict, X)
            shap_values = explainer.shap_values(X)
        
        return shap_values
    
    def save_model(self, filepath):
        """Save the trained model and associated data"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'features_list': self.features_list,
            'hyperparameters': self.hyperparameters,
            'threshold': self.threshold
        }
        joblib.dump(model_data, filepath)
        logging.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        predictor = cls()
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.features_list = model_data['features_list']
        predictor.hyperparameters = model_data['hyperparameters']
        predictor.threshold = model_data['threshold']
        return predictor
    
    def _get_model_instance(self, model_type):
        """Helper method to get model instance based on type"""
        if model_type == 'Random Forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(random_state=42)
        elif model_type == 'Gradient Boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(random_state=42)
        elif model_type == 'Logistic Regression':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(random_state=42)
        elif model_type == 'SVM':
            from sklearn.svm import SVC
            return SVC(probability=True, random_state=42)
        elif model_type == 'XGBoost':
            import xgboost as xgb
            return xgb.XGBClassifier(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")