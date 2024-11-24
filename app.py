from flask import Flask, request, jsonify
import pandas as pd
import os
import logging
from nba_career_predictor import NBACareerPredictor
import config
from werkzeug.utils import secure_filename

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
        
        # Scale the features using the fitted scaler
        enhanced_df_scaled = pd.DataFrame(
            predictor.scaler.transform(enhanced_df),
            columns=enhanced_df.columns
        )
        
        probability = predictor.predict_proba(enhanced_df_scaled)[0]
        prediction = int(probability >= predictor.threshold)
        
        response = {
            'prediction': prediction,
            'probabilite': round(float(probability),2),
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)