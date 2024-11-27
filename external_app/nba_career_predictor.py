import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import joblib
import logging
import config

class NBACareerPredictor:
    def __init__(self, model_type="Random Forest", seed=42):
        self.model = None
        self.model_type = model_type
        self.features_list = None
        self.scaler = MinMaxScaler()
        self.hyperparameters = {}
        self.feature_names = None
        self.threshold = config.PREDICT_THRESHOLD
        self.seed = seed
        
    def add_features(self, df):
        """Add engineered features to the dataset"""
        logging.info("Adding engineered features...")
        
        # Create copy to avoid modifying original
        enhanced_df = df.copy()
        
        # Total features
        for col in df.columns:
            if col not in ['Name', config.OUTCOME, 'FG%', '3P%', 'FT%', 'GP']:
                enhanced_df[f'{col}_TOT'] = df['GP'] * df[col]
        
        # Efficiency rates
        tot_cols = [col for col in enhanced_df.columns if ((col.endswith('_TOT')) and (col != 'MIN_TOT'))]
        for col in tot_cols:
            base_name = col.replace('_TOT', '')
            enhanced_df[f'{base_name}_RATE'] = enhanced_df[col] / np.maximum(enhanced_df['MIN_TOT'], 1)
        # drop all tot_cols
        enhanced_df = enhanced_df.drop(tot_cols, axis=1)
        
        # Categorical features for minutes and games
        enhanced_df['MIN_CAT'] = pd.cut(
            enhanced_df['MIN'],
            bins=config.MINUTES_BINS,
            labels=['MIN<10', 'MIN<20', 'MIN<30', 'MIN>30']
        )
        enhanced_df['GP_CAT'] = pd.cut(
            enhanced_df['GP'],
            bins=config.GAMES_BINS,
            labels=['GP<28', 'GP<56', 'GP≥56']
        )
        
        # One-hot encode categorical features
        enhanced_df = pd.get_dummies(enhanced_df, columns=['MIN_CAT', 'GP_CAT'])
        
        # add Two random variables: RANDOM_BINARY and RANDOM_NUMERICAL
        if config.ADD_RANDOM_VARIABLES:
            np.random.seed(42)
            enhanced_df['RANDOM_BINARY'] = np.random.randint(0, 2, size=len(enhanced_df))
            enhanced_df['RANDOM_NUMERICAL'] = np.random.randint(0, 1000, size=len(enhanced_df))
        
        logging.info(f"Added {len(enhanced_df.columns) - len(df.columns)} new features")
        return enhanced_df
    
    def preprocess_data(self, df):
        """Preprocess the dataset"""
        logging.info("Preprocessing data...")
        
        # Replace NaN values
        df = df.fillna(0)
        
        # Drop duplicates
        df = df.groupby([col for col in df.columns if col != config.OUTCOME])\
               .agg({config.OUTCOME: 'max'})\
               .reset_index()
        
        logging.info(f"Dataset contains {df['Name'].nunique()} unique players")
        
        # Store feature names
        self.features_list = [col for col in df.columns if col not in ['Name', config.OUTCOME]]
        
        return df
    
    def hyperparameter_tuning_model(self, X_train, y_train):
        """Tuning hyperparameters for the model"""
        model_type = self.model_type
        logging.info(f"Fine-tuning {model_type} model...")
        
        # Get base model and parameter grid
        base_model = self._get_model_instance()
        param_grid = config.PARAM_GRIDS[model_type]
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=self.seed)
        
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
    
    def train_model(self, X_train, y_train):
        """Train the model"""
        logging.info("Training model...")
        # # Store feature names
        # self.features_list = [col for col in X_train.columns if col not in ['Name', OUTCOME]]
        
        # Get tuned model
        self.model = self.hyperparameter_tuning_model(X_train, y_train)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make training predictions
        y_pred = self.model.predict(X_train)
        y_prob = self.model.predict_proba(X_train)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_train, y_pred),
            'precision': precision_score(y_train, y_pred),
            'recall': recall_score(y_train, y_pred),
            'f1': f1_score(y_train, y_pred)
        }
        
        final_score = metrics[config.GOAL_METRIC]
        
        return metrics, final_score
    
    def test_model(self, X_test, y_test):
        """Test the model and calculate metrics"""
        logging.info("Testing model...")
        
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
        
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)
        metrics['auc'] = auc_score
        
        # Calculate Youden Index
        youden_index, optimal_threshold, optimal_fpr, optimal_tpr = self.calculate_youden_index(
            fpr, tpr, thresholds
        )
        
        logging.info(f"Model performance metrics (AUC={auc_score:.3f}):")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.3f}")
        
        return metrics, fpr, tpr, thresholds, youden_index, optimal_threshold, optimal_fpr, optimal_tpr
    
    def calculate_youden_index(self, fpr, tpr, thresholds):
        """Calculate Youden Index and find optimal threshold"""
        youden_index = tpr - fpr
        max_index = np.argmax(youden_index)
        return youden_index[max_index], thresholds[max_index], fpr[max_index], tpr[max_index]
    
    def predict_proba(self, X):
        """Predict probability for multiple samples"""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        # Ensure all required features are present
        missing_features = set(self.features_list) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select and order features
        X = X[self.features_list]
        
        # Scale features using the fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save_model(self, filepath):
        """Save the trained model and associated data"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'features_list': self.features_list,
            'hyperparameters': self.hyperparameters,
            'threshold': self.threshold
        }
        print(f"Model data threshold: {round(model_data['threshold'],2)}")
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
    
    def _get_model_instance(self):
        """Helper method to get model instance based on type"""
        model_type = self.model_type
        if model_type == 'Random Forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(random_state=self.seed)
        elif model_type == 'Gradient Boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(random_state=self.seed)
        elif model_type == 'Logistic Regression':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(random_state=self.seed)
        elif model_type == 'SVM':
            from sklearn.svm import SVC
            return SVC(probability=True, random_state=self.seed)
        elif model_type == 'XGBoost':
            import xgboost as xgb
            return xgb.XGBClassifier(random_state=self.seed)
        else:
            raise ValueError(f"Unknown model type: {model_type}")