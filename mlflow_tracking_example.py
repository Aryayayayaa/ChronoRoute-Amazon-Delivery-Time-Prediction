import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# --- MOCK DATA AND MODEL (Replace with your actual data and model object) ---
# Assuming these variables exist after training
X_test_mock = np.random.rand(10, 5) 
y_test_mock = np.random.rand(10) * 100
optimized_rf_model_mock = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
optimized_rf_model_mock.fit(np.random.rand(100, 5), np.random.rand(100) * 100) # Simple fit for demo
# --- END MOCK ---

# 1. Define the Experiment Name
MLFLOW_EXPERIMENT_NAME = "Delivery_Time_Prediction_Comparison"

# Check if experiment exists, otherwise create it
if not mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME):
    mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# 2. Start MLflow Run for the Best Model
with mlflow.start_run(run_name="Optimized_Random_Forest_Final_Model") as run:
    
    # Calculate performance metrics
    predictions = optimized_rf_model_mock.predict(X_test_mock)
    rmse = sqrt(mean_squared_error(y_test_mock, predictions))
    r2_score = optimized_rf_model_mock.score(X_test_mock, y_test_mock)
    
    # 3. Log Parameters (Hyperparameters used for the Optimized RF)
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("optimization_method", "RandomizedSearchCV")
    
    # Log the specific hyperparameters that were found to be optimal
    optimal_params = {
        "n_estimators": 200,    # Example: optimal trees
        "max_depth": 18,        # Example: optimal depth
        "min_samples_split": 5  # Example: optimal split
    }
    mlflow.log_params(optimal_params)
    
    # 4. Log Metrics (Performance scores)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2_Score", r2_score)
    mlflow.log_metric("Average_Error_Mins", 22.3785) # Logging the verified score
    
    # 5. Log the Model Artifact (the actual joblib file)
    # Note: MLflow will log the model file and metadata, making it retrievable later.
    mlflow.sklearn.log_model(
        sk_model=optimized_rf_model_mock, 
        artifact_path="model", 
        registered_model_name="Delivery_Predictor_RF_Optimized"
    )

    print(f"MLflow Run ID: {run.info.run_id}")
    print("Model logged successfully to MLflow tracking server.")


