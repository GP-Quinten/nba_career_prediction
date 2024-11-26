from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import os
import logging
from nba_career_predictor import NBACareerPredictor
import config
from werkzeug.utils import secure_filename
import shap
import base64
from io import BytesIO
# Add these imports at the top
import matplotlib
matplotlib.use('Agg')  # Set this BEFORE importing pyplot
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'temp_uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = os.path.join(config.MODELS_DIR, 'production_model', 'final_model.joblib')
predictor = NBACareerPredictor.load_model(MODEL_PATH)

def read_excel_input(file):
    """Read excel file and return player stats dictionary"""
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        df = pd.read_excel(filepath, index_col=0)
        os.remove(filepath)  # Clean up
        return df.iloc[0].to_dict()
    except Exception as e:
        logger.error(f"Error reading Excel file: {str(e)}")
        raise

def get_shap_explanation(model, enhanced_df_scaled, feature_names):
    """Generate SHAP summary plot"""
    try:
        logger.info("Starting SHAP explanation generation...")
        plt.clf()
        
        from sklearn.linear_model import LogisticRegression
        if isinstance(model, LogisticRegression):
            logger.info("Using direct coefficient interpretation for Logistic Regression")
            
            # Get coefficients and feature values
            coef = model.coef_[0]  # Coefficients for binary classification
            feature_values = enhanced_df_scaled.values[0]
            
            # Calculate feature contributions
            contributions = coef * feature_values
            
            logger.info(f"Coefficients shape: {coef.shape}")
            logger.info(f"Feature values shape: {feature_values.shape}")
            logger.info(f"Contributions shape: {contributions.shape}")
            logger.info(f"Sample contributions: {contributions[:5]}")
            
            # Get top 10 most important features
            importance = np.abs(contributions)
            top_indices = np.argsort(importance)[-10:]
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            # Plot horizontal bar chart
            y_pos = np.arange(len(top_indices))
            values_to_plot = contributions[top_indices]
            feature_names_to_plot = np.array(feature_names)[top_indices]
            
            bars = plt.barh(y_pos, values_to_plot)
            
            # Color bars based on contribution direction
            for i, bar in enumerate(bars):
                bar.set_color('red' if values_to_plot[i] > 0 else 'blue')
            
            # Add feature names and values
            plt.yticks(y_pos, feature_names_to_plot)
            for i, v in enumerate(values_to_plot):
                plt.text(v, i, f'{v:.3f}', va='center')
            
            plt.xlabel('Feature Contribution (coefficient × value)')
            plt.title('Top 10 Feature Contributions to Prediction')
            
            plt.tight_layout()
            
        # Convert plot to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        logger.info("Feature importance visualization generated successfully")
        return plot_base64
        
    except Exception as e:
        logger.error(f"Error in get_shap_explanation: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Received prediction request")
        if 'file' in request.files:
            file = request.files['file']
            logger.info(f"Processing file: {file.filename}")
            if file and file.filename.endswith('.xlsx'):
                player_stats = read_excel_input(file)
            else:
                return jsonify({'error': 'Invalid file format. Please upload .xlsx file'}), 400
        else:
            player_stats = request.get_json()
            logger.info("Processing JSON input")

        if not player_stats:
            return jsonify({'error': 'Aucune donnée fournie'}), 400
            
        required_features = set([
            'GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3PA', '3P%',
            'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV'
        ])
        
        missing_features = required_features - set(player_stats.keys())
        if missing_features:
            return jsonify({
                'error': f'Features manquantes: {missing_features}'
            }), 400
        
        input_df = pd.DataFrame([player_stats])
        enhanced_df = predictor.add_features(input_df)
        
        # Scale the features
        enhanced_df_scaled = pd.DataFrame(
            predictor.scaler.transform(enhanced_df),
            columns=enhanced_df.columns
        )
        
        # Get prediction
        probability = predictor.predict_proba(enhanced_df_scaled)[0]
        prediction = int(probability >= predictor.threshold)
        
        # Generate SHAP explanation
        shap_plot = get_shap_explanation(predictor.model, enhanced_df_scaled, enhanced_df.columns)
        
        # Store results for visualization
        app.shap_results = {
            'prediction': prediction,
            'probability': round(float(probability),2),
            'shap_plot': shap_plot
        }
        
        if config.API == 'internal':
            response = {
                'prediction': prediction,
                'probabilite': round(float(probability),2),
                'message': 'View explanation at http://127.0.0.1:5000/explain'
            }
        else:
            response = {
                'prediction': prediction,
                'probabilite': round(float(probability),2),
                'message': f'View explanation at {request.headers.get("X-Forwarded-Proto", "http")}://{request.headers.get("Host")}/explain'
            }
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/explain', methods=['GET'])
def explain():
    """Display the SHAP explanation for the last prediction"""
    try:
        if not hasattr(app, 'shap_results'):
            return "No prediction available to explain", 404
            
        return render_template_string(
            config.HTML_SHAP_LOCAL_EXPL_TEMPLATE,
            prediction=app.shap_results['prediction'],
            probability=app.shap_results['probability'],
            shap_plot=app.shap_results['shap_plot']
        )
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        return str(e), 500

if __name__ == '__main__':
    if config.API == 'external':
        app.run(debug=True, port=5000, host='0.0.0.0')
    else:
        app.run(debug=True, port=5000, host='127.0.0.1')