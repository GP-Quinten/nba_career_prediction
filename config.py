import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# File paths
DATA_PATH = os.path.join(DATA_DIR, 'nba_logreg.csv')
METRICS_PATH = os.path.join(RESULTS_DIR, 'metrics.csv')
ROC_PLOT_PATH = os.path.join(RESULTS_DIR, 'roc_curves.html')

# Model configurations
RANDOM_SEED = 42
GOAL_METRIC = 'auprc'  # alternatively 'recall'
N_SPLITS = 5
XAI = True  # Whether to compute SHAP values

# Model parameter grids
PARAM_GRIDS = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', None]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    },
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'class_weight': ['balanced', None],
        'solver': ['lbfgs', 'liblinear']
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'class_weight': ['balanced', None]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
}

# Feature engineering configurations
MINUTES_BINS = [0, 10, 20, 30, float('inf')]
GAMES_BINS = [0, 28, 56, float('inf')]
OUTCOME = 'TARGET_5Yrs'