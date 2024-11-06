import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import os
from nba_career_predictor import NBACareerPredictor
from parser import setup_parser
import config
import shap
from sklearn.metrics import roc_curve, auc

def main():
    # Parse arguments
    args = setup_parser()
    
    # Create experiment directory
    experiment_dir = os.path.join(config.RESULTS_DIR, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize results tracking
    metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1'])
    fig = go.Figure()
    
    # Load data
    logging.info("Loading data...")
    df = pd.read_csv(config.DATA_PATH)
    
    # Initialize predictor
    predictor = NBACareerPredictor()
    
    # Add features
    logging.info("Adding features...")
    enhanced_df = predictor.add_features(df)
    
    # Save enhanced dataset
    enhanced_df.to_csv(os.path.join(experiment_dir, 'enhanced_data.csv'), index=False)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = predictor.preprocess_data(enhanced_df)
    
    # Save train and test sets
    np.save(os.path.join(experiment_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(experiment_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(experiment_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(experiment_dir, 'y_test.npy'), y_test)
    
    # Train and evaluate each model
    for model_name in args.models:
        logging.info(f"\nProcessing {model_name}...")
        
        # Train and evaluate model
        metrics, conf_matrix, y_prob = predictor.train_and_test_model(
            X_train, y_train, X_test, y_test, model_name
        )
        
        # Save model and results
        model_dir = os.path.join(experiment_dir, model_name.replace(' ', '_'))
        os.makedirs(model_dir, exist_ok=True)
        predictor.save_model(os.path.join(model_dir, 'model.joblib'))
        
        # Add metrics to the table
        metrics_df = metrics_df.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1']
        }, ignore_index=True)
        
        # Plot ROC curve for model
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_name} (AUC={auc_score:.2f})'))
        
        # Explain predictions if XAI is enabled
        if config.XAI:
            shap_values = predictor.explain_predictions(X_test)
            shap.summary_plot(shap_values, X_test, feature_names=predictor.features_list)
    
    # Save metrics and ROC plot
    metrics_df.to_csv(os.path.join(experiment_dir, 'metrics.csv'), index=False)
    fig.update_layout(title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    fig.write_html(os.path.join(experiment_dir, 'roc_curves.html'))
    
    # Display metrics table and ROC curves plot
    logging.info("Final metrics table:\n" + str(metrics_df))
    fig.show()

if __name__ == '__main__':
    main()
