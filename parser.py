import argparse
import logging

VALID_MODELS = ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 'SVM', 'XGBoost']

def setup_parser():
    parser = argparse.ArgumentParser(description='NBA Career Prediction Model Training')
    
    parser.add_argument(
        '--models',
        nargs='+',
        help=f'List of models to train. Valid options: {VALID_MODELS}',
        required=True
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='experiment',
        help='Name of the experiment for saving results'
    )

    parser.add_argument(
        '--add-random-variable',
        choices=[True, False],
        type=bool,
        default=False,
        help='You can add a random variable to the dataset to test the coherence of the model'
    )
    
    args = parser.parse_args()
    
    # Validate models
    invalid_models = [model for model in args.models if model not in VALID_MODELS]
    if invalid_models:
        parser.error(f"Invalid models: {invalid_models}. Valid options are: {VALID_MODELS}")
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    return args

if __name__ == '__main__':
    args = setup_parser()
    print(f"Selected models: {args.models}")
    print(f"Log level: {args.log_level}")
    print(f"Experiment name: {args.experiment_name}")
