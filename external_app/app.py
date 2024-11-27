# Standard library imports
import os
import sys
from pathlib import Path
from io import BytesIO
import base64
import logging

# Third-party imports
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use("Agg")  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Local imports
from nba_career_predictor import NBACareerPredictor
import config
from logger import init_logger

# Initialize paths
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "temp_uploads"
LOG_FILE = BASE_DIR / "api.log"

# Initialize logging
init_logger(
    level="INFO",
    file=True,
    file_path=str(LOG_FILE),  # Convert Path to string for logger
    stream=True
)

# Flask application setup
app = Flask(__name__)

# Configure Flask application logging
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.DEBUG)

app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    UPLOAD_FOLDER=str(UPLOAD_FOLDER)
)

# Ensure required directories exist
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Load the model
try:
    predictor = NBACareerPredictor.load_model(config.MODEL_PATH)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

def read_excel_input(file):
    """Read excel file and return player stats dictionary"""
    try:
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename  # Use pathlib for path joining
        file.save(str(filepath))
        df = pd.read_excel(filepath, index_col=0)
        filepath.unlink()  # Delete file using pathlib
        return df.iloc[0].to_dict()
    except Exception as e:
        logging.error(f"Error reading Excel file: {str(e)}")
        raise

def get_feature_importance_plot(model, enhanced_df_scaled, feature_names):
    """Generate feature importance visualization"""
    try:
        logging.info("Starting feature importance visualization...")
        plt.clf()

        if isinstance(model, LogisticRegression):
            # Calculate feature contributions
            coef = model.coef_[0]
            feature_values = enhanced_df_scaled.values[0]
            contributions = coef * feature_values
            
            # Get top 10 features
            importance = np.abs(contributions)
            top_indices = np.argsort(importance)[-10:]
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            y_pos = np.arange(len(top_indices))
            values_to_plot = contributions[top_indices]
            feature_names_to_plot = np.array(feature_names)[top_indices]
            
            # Plot bars
            bars = plt.barh(y_pos, values_to_plot)
            for i, bar in enumerate(bars):
                bar.set_color("red" if values_to_plot[i] > 0 else "blue")
                plt.text(values_to_plot[i], i, f"{values_to_plot[i]:.3f}", va="center")
            
            # Customize plot
            plt.yticks(y_pos, feature_names_to_plot)
            plt.xlabel("Feature Contribution")
            plt.title("Top 10 Feature Contributions to Prediction")
            plt.tight_layout()
            
            # Convert to base64
            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logging.error(f"Error in feature importance visualization: {str(e)}")
        raise

def get_explain_url(request):
    """Generate appropriate URL for explanation endpoint"""
    scheme = request.headers.get("X-Forwarded-Proto", request.scheme)
    host = request.headers.get("X-Forwarded-Host", request.host)
    return f"{scheme}://{host}/explain"

# Routes
@app.route("/", methods=["GET"])
def home():
    """Health check endpoint"""
    return "NBA Career Predictor API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    """Make prediction and generate feature importance visualization"""
    try:
        logging.info("Received prediction request")
        
        # Handle input data
        if "file" in request.files:
            file = request.files["file"]
            if not file.filename.endswith(".xlsx"):
                return jsonify({"error": "Invalid file format. Please upload .xlsx file"}), 400
            player_stats = read_excel_input(file)
        else:
            player_stats = request.get_json()
            if not player_stats:
                return jsonify({"error": "No data provided"}), 400

        # Validate features
        required_features = {
            "GP", "MIN", "PTS", "FGM", "FGA", "FG%", "3P Made", "3PA", "3P%",
            "FTM", "FTA", "FT%", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV"
        }
        if missing := (required_features - set(player_stats.keys())):
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # Process data and make prediction
        input_df = pd.DataFrame([player_stats])
        enhanced_df = predictor.add_features(input_df)
        enhanced_df_scaled = pd.DataFrame(
            predictor.scaler.transform(enhanced_df),
            columns=enhanced_df.columns
        )
        
        probability = predictor.predict_proba(enhanced_df_scaled)[0]
        prediction = int(probability >= predictor.threshold)
        
        # Generate visualization
        importance_plot = get_feature_importance_plot(
            predictor.model,
            enhanced_df_scaled,
            enhanced_df.columns
        )
        
        # Store results for explanation endpoint
        app.visualization_results = {
            "prediction": prediction,
            "probability": round(float(probability), 2),
            "importance_plot": importance_plot
        }
        
        return jsonify({
            "prediction": prediction,
            "probabilite": round(float(probability), 2),
            "message": f"View explanation at {get_explain_url(request)}"
        })

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/explain", methods=["GET"])
def explain():
    """Display feature importance visualization"""
    try:
        if not hasattr(app, "visualization_results"):
            return "No prediction available to explain", 404

        return render_template_string(
            config.HTML_SHAP_LOCAL_EXPL_TEMPLATE,
            prediction=app.visualization_results["prediction"],
            probability=app.visualization_results["probability"],
            shap_plot=app.visualization_results["importance_plot"]
        )
    except Exception as e:
        logging.error(f"Error generating explanation: {str(e)}")
        return str(e), 500

# Main entry point
if __name__ == "__main__":
    # Add stdout logging for development
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.DEBUG)
    
    # Run the application
    if config.API == "external":
        app.run(debug=True, port=5000, host="0.0.0.0")
    else:
        app.run(debug=True, port=5000, host="127.0.0.1")