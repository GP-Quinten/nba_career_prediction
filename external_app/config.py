from pathlib import Path

# Base paths - all files are now relative to the root directory
BASE_DIR = Path(__file__).resolve().parent

# We can remove MODELS_DIR since the model is at root level
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'

# Create directories if they don't exist - note we removed MODELS_DIR
for dir_path in [DATA_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# File paths
DATA_PATH = DATA_DIR / 'nba_logreg.csv'
METRICS_PATH = RESULTS_DIR / 'metrics.csv'
ROC_PLOT_PATH = RESULTS_DIR / 'roc_curves.html'
MODEL_PATH = BASE_DIR / 'final_model.joblib'  # Model is now at root level
# Model configurations
RANDOM_SEED = 10
GOAL_METRIC = 'recall'  # alternatively 'recall'
N_SPLITS = 10
XAI = False  # Whether to compute SHAP values
PREDICT_THRESHOLD = 1/3 # Threshold for predicting positive class
ADD_RANDOM_VARIABLES = False

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
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [10000]
    },
    'SVM': {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'class_weight': ['balanced', None],
        'probability': [True]
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

# parameters for plotting
MODELS_COLORS = {
    "Random Forest": "blue",
    "Gradient Boosting": "green",
    "Logistic Regression": "red",
    "SVM": "orange",
    "XGBoost": "purple"
}

# API Link
API = "external" # internal or external

# parameters for app SHAP local explanation HTML template
HTML_FEATURE_IMPORTANCE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SHAP Explanation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .result {
            margin: 20px 0;
            padding: 20px;
            background-color: #f5f5f5;
            border-radius: 5px;
        }
        .probability {
            font-weight: bold;
            color: #2c3e50;
        }
        img {
            max-width: 100%;
            height: auto;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h2>Prediction Results</h2>
    <div class="result">
        <p>Prediction: <span class="probability">{{ 'Carrière > 5 ans' if prediction == 1 else 'Carrière <= 5 ans' }}</span></p>
        <p>Probabilité: <span class="probability">{{ "%.2f%%" | format(probability * 100) }}</span></p>
    </div>
    <h3>SHAP Explanation</h3>
    <img src="data:image/png;base64,{{ shap_plot }}" />
</body>
</html>
"""
