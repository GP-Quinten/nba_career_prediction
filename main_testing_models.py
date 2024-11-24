import pandas as pd
import plotly.graph_objects as go
import logging
import os
from sklearn.model_selection import train_test_split
from nba_career_predictor import NBACareerPredictor
from parser import setup_parser
import config
import shap
import json
import matplotlib.pyplot as plt

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parse arguments
    args = setup_parser()
    
    # Create experiment directory
    experiment_dir = os.path.join(config.RESULTS_DIR, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize results tracking
    metrics_df = pd.DataFrame({
        'Model': pd.Series(dtype='str'),
        'Accuracy': pd.Series(dtype='float'),
        'Precision': pd.Series(dtype='float'),
        'Recall': pd.Series(dtype='float'),
        'F1': pd.Series(dtype='float'),
        'AUC': pd.Series(dtype='float')
    })
    
    # Initialize ROC curve plot
    fig = go.Figure()
    
    # Load data
    logging.info("Loading data...")
    df = pd.read_csv(config.DATA_PATH)
    
    # Initialize base predictor for feature engineering and preprocessing
    base_predictor = NBACareerPredictor(seed=args.seed)
    
    # Add features
    logging.info("Adding engineered features...")
    enhanced_df = base_predictor.add_features(df)
    
    # Preprocess data
    logging.info("Preprocessing data...")
    processed_df = base_predictor.preprocess_data(enhanced_df)
    
    # Split features and target
    X = processed_df[base_predictor.features_list]
    y = processed_df[config.OUTCOME]
    
    # Split data
    logging.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    
    # Save enhanced and processed datasets
    enhanced_df.to_csv(os.path.join(experiment_dir, 'enhanced_data.csv'), index=False)
    processed_df.to_csv(os.path.join(experiment_dir, 'processed_data.csv'), index=False)
    
    # Save train and test sets
    pd.DataFrame(X_train, columns=base_predictor.features_list).to_csv(
        os.path.join(experiment_dir, 'X_train.csv'), index=False)
    pd.DataFrame(X_test, columns=base_predictor.features_list).to_csv(
        os.path.join(experiment_dir, 'X_test.csv'), index=False)
    pd.Series(y_train).to_csv(os.path.join(experiment_dir, 'y_train.csv'), index=False)
    pd.Series(y_test).to_csv(os.path.join(experiment_dir, 'y_test.csv'), index=False)
    
    # Train and evaluate each model
    predictors_dict = {}
    for model_name in args.models:
        logging.info("=" * 50)
        logging.info(f"Processing {model_name}...")
        logging.info("=" * 50)
        
        # Initialize model-specific predictor
        predictors_dict[model_name] = NBACareerPredictor(model_type=model_name, seed=args.seed)
        predictor = predictors_dict[model_name]
        
        # Fit scaler on training data and transform both sets
        X_train_scaled = predictor.scaler.fit_transform(X_train)
        X_test_scaled = predictor.scaler.transform(X_test)
        
        # Train model
        metrics, final_score = predictor.train_model(X_train_scaled, y_train)
        
        # Test model
        test_metrics, fpr, tpr, thresholds, youden_index, optimal_threshold, optimal_fpr, optimal_tpr = (
            predictor.test_model(X_test_scaled, y_test)
        )
        
        # Save model, hyperparameters and results
        model_dir = os.path.join(experiment_dir, model_name.replace(' ', '_'))
        os.makedirs(model_dir, exist_ok=True)
        predictor.save_model(os.path.join(model_dir, 'model.joblib'))
        # Save model metadata
        metadata = {
            'model_type': predictor.model_type,
            'features': base_predictor.features_list,
            'hyperparameters': predictor.hyperparameters,
            'metrics': metrics,
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'number_of_features': len(base_predictor.features_list),
            'training_samples': len(X),
            'positive_class_ratio': float(y.mean()),
            'random_seed': args.seed
        }
        
        # Save metadata as JSON
        with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

        # Add metrics to the results DataFrame
        new_metrics_row = pd.DataFrame([{
            'Model': model_name,
            'Accuracy': test_metrics['accuracy'],
            'Precision': test_metrics['precision'],
            'Recall': test_metrics['recall'],
            'F1': test_metrics['f1'],
            'AUC': test_metrics['auc']
        }])
        metrics_df = pd.concat([metrics_df, new_metrics_row], ignore_index=True)
        
        # Add ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            line=dict(color=config.MODELS_COLORS[model_name]),
            name=f"{model_name} (AUC={test_metrics['auc']:.3f})",
            hovertemplate=(
                "FPR: %{x:.3f}<br>"
                "TPR: %{y:.3f}<br>"
                "Threshold: %{customdata:.3f}"
                "<extra></extra>"
            ),
            customdata=thresholds
        ))
        
        # Add optimal threshold point
        fig.add_trace(go.Scatter(
            x=[optimal_fpr], y=[optimal_tpr],
            mode='markers',
            marker=dict(color='red', size=8),
            name=f'{model_name} Optimal (thresh={optimal_threshold:.2f})',
            hovertemplate=(
                "Optimal threshold: %{text:.3f}<br>"
                "FPR: %{x:.3f}<br>"
                "TPR: %{y:.3f}"
                "<extra></extra>"
            ),
            text=[optimal_threshold]
        ))
        
        # Generate SHAP values if enabled
        if config.XAI:
            logging.info(f"Generating SHAP values for {model_name}...")
            shap_values = predictor.explain_predictions(X_test_scaled)
            
            # Create SHAP summary plot
            plt.figure()
            shap.summary_plot(
                shap_values, 
                X_test_scaled,
                feature_names=predictor.features_list,
                show=False
            )
            plt.title(f'SHAP Summary Plot - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, 'shap_summary.png'))
            plt.close()
    
    # Save final results
    metrics_df.to_csv(os.path.join(experiment_dir, 'metrics.csv'), index=False)
    
    # Update and save ROC plot
    fig.update_layout(
        title=f"ROC Curves Comparison - {args.experiment_name}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=900,
        height=600,
        showlegend=True
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.write_html(os.path.join(experiment_dir, 'roc_curves.html'))
    fig.write_image(os.path.join(experiment_dir, 'roc_curves.png'))
    
    # Display final metrics
    logging.info("\nFinal metrics comparison:")
    logging.info("\n" + str(metrics_df.to_string()))
    
    # Find best model based on goal metric
    best_model = metrics_df.loc[metrics_df[config.GOAL_METRIC.title()].idxmax(), 'Model']
    logging.info(f"\nBest model based on {config.GOAL_METRIC}: {best_model}")

if __name__ == '__main__':
    main()