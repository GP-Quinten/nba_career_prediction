import config
import logging
import os
from parser import setup_parser
import pandas as pd
import numpy as np
import logging
import os
from nba_career_predictor import NBACareerPredictor
import config
import shap
import matplotlib.pyplot as plt

def main():
    # Parse arguments
    args = setup_parser()
    
    if not args.final_model:
        raise ValueError("Please use --final-model flag for training the production model")
    
    if len(args.models) != 1:
        raise ValueError("Please specify exactly one model for final training")
    
    model_type = args.models[0]
    
    # Create output directory
    output_dir = os.path.join(config.MODELS_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Add file handler for logging
    fh = logging.FileHandler(os.path.join(output_dir, 'model_training.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(fh)

    # Load data
    logging.info("Loading data...")
    df = pd.read_csv(config.DATA_PATH)
    logging.info(f"Loaded {len(df)} players data")
    
    # Initialize predictor
    predictor = NBACareerPredictor(model_type=model_type, seed=args.seed)
    
    # Add features
    logging.info("Engineering features...")
    enhanced_df = predictor.add_features(df)
    
    # Save feature names and their descriptions
    feature_descriptions = pd.DataFrame({
        'Feature': enhanced_df.columns,
        'Type': enhanced_df.dtypes,
        'Has_Missing': enhanced_df.isnull().any(),
        'Unique_Values': [len(enhanced_df[col].unique()) for col in enhanced_df.columns]
    })
    feature_descriptions.to_csv(os.path.join(output_dir, 'feature_descriptions.csv'), index=False)
    
    # Preprocess data
    logging.info("Preprocessing data...")
    processed_df = predictor.preprocess_data(enhanced_df)
    
    # Split features and target
    X = processed_df[predictor.features_list]
    y = processed_df[config.OUTCOME]
    
    # Scale all data
    logging.info("Scaling features...")
    X_scaled = predictor.scaler.fit_transform(X)
    
    # Train final model
    logging.info(f"Training final {model_type} model...")
    metrics, final_score = predictor.train_model(X_scaled, y)
    
    # Log training results
    logging.info("\nTraining metrics:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.3f}")
    
    # Generate feature importance plot if applicable
    if hasattr(predictor.model, 'feature_importances_'):
        logging.info("Generating feature importance plot...")
        
        # Get feature importances
        importances = predictor.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.title('Feature Importances')
        plt.bar(range(X.shape[1]), importances[indices])
        plt.xticks(range(X.shape[1]), [predictor.features_list[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importances.png'))
        plt.close()
        
        # Save feature importances to CSV
        importance_df = pd.DataFrame({
            'Feature': [predictor.features_list[i] for i in indices],
            'Importance': importances[indices]
        })
        importance_df.to_csv(os.path.join(output_dir, 'feature_importances.csv'), index=False)
    
    # Generate SHAP values if enabled
    logging.info("Generating SHAP values...")
    shap_values = predictor.explain_predictions(X_scaled)
    
    # Create and save SHAP summary plot
    plt.figure()
    shap.summary_plot(
        shap_values, 
        X_scaled,
        feature_names=predictor.features_list,
        show=False
    )
    plt.title('SHAP Values - Feature Impact')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
    plt.close()
    
    # Save model and metadata
    logging.info("Saving model and metadata...")
    
    # Save the model
    predictor.save_model(os.path.join(output_dir, 'final_model.joblib'))
    
    # Save model metadata
    metadata = {
        'model_type': model_type,
        'features': predictor.features_list,
        'hyperparameters': predictor.hyperparameters,
        'metrics': metrics,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'number_of_features': len(predictor.features_list),
        'training_samples': len(X),
        'positive_class_ratio': float(y.mean()),
        'random_seed': args.seed
    }
    
    # Save metadata as JSON
    import json
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Create a simple example prediction file
    example_input = pd.DataFrame([{
        col: 0 for col in predictor.features_list
    }])
    example_input.to_csv(os.path.join(output_dir, 'example_input.csv'), index=False)
    
    logging.info(f"Model and metadata saved to {output_dir}")
    logging.info("Training complete!")

if __name__ == '__main__':
    main()