# NBA Career Predictor

## Usage for external user

### Web Deployment on Free PythonAnyWhere
The application is also deployed on PythonAnywhere. To use it:

1. Download the template `input.xlsx` and fill the stats of your player accordingly
2. Make predictions using:
```bash
curl -X POST -F "file=@/path/to/your/input.xlsx" https://Gagatiu.pythonanywhere.com/predict
```
3. Open the link of explanation to understand how the different features influenced the prediction.

### Model Details
- Threshold set to 1/3 to achieve recall > 80%. Recall is prioritized to avoid missing potential key players (like the next Stephen Curry).
A false positive rate of 80% is acceptable (1 out of 5 predictions should be a key player).
- Feature importance based on logistic regression coefficients (simplified interpretation due to hosting constraints)
- The API response includes:
  - Prediction (career > 5 years: yes/no)
  - Probability score
  - Link to visualize feature importance analysis

## Structure and Usage

### Exploratory Analysis
You can find Jupyter notebooks with exploratory data analysis in the `notebooks/` folder.

### Model Selection
Run model comparison script (`main_testing_models.py`) to evaluate different models:
```bash
python main_testing_models.py \
    --models "Random Forest" "Logistic Regression" "Gradient Boosting" "SVM" "XGBoost" \
    --experiment-name "model_comparison" \
    --seed 42 \
    --log-level INFO \
```
You will find some info on the experiment in the `results/model_comparison`. Namely check the plot of the ROC curves of all models.

### Final Model Training
Once you've selected your model (we chose Logistic Regression), train it on the complete dataset:
```bash
python main_train_final_model.py \
    --models "Logistic Regression" \
    --log-level INFO \
    --seed 42 \
    --output-dir "production_model" \
    --final-model
```

### Local API Usage
The `app.py` script provides a local API interface:

1. Run the server:
```bash
python app.py
```

2. Make predictions using the provided Excel template (`input.xlsx`):
```bash
curl -X POST -F "file=@inputs/input_1.xlsx" http://127.0.0.1:5000/predict
```

Note: The input Excel file should contain two rows:
- Row 1: Headers with required player statistics
- Row 2: Corresponding values for your player


### Directory Structure
All files needed for deployment are gathered in the `external_app/` folder, which contains the files uploaded to PythonAnywhere.
