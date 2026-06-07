#Part 1: Register Models
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

# Set tracking server endpoint
mlflow.set_tracking_uri("http://localhost:5000")

# Initialize the MLflow Tracking API Client
client = MlflowClient()

# Retrieve experiment metadata
experiment = client.get_experiment_by_name("predictive-maintenance")
print(f"Experiment ID: {experiment.experiment_id}")
#import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

# Set tracking server endpoint
mlflow.set_tracking_uri("http://localhost:5000")

# Initialize the MLflow Tracking API Client
client = MlflowClient()

# Retrieve experiment metadata
experiment = client.get_experiment_by_name("predictive-maintenance")
print(f"Experiment ID: {experiment.experiment_id}")

#Task 1.2: View Existing Runs

# Query all runs from the experiment, ordered by descending ROC AUC score
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.roc_auc DESC"],
)

# Parse relevant structural metadata
run_data = []
for run in runs:
    run_data.append(
        {
            "run_id": run.info.run_id[:8],
            "name": run.info.run_name,
            "model_type": run.data.params.get("model_type", "Unknown"),
            "roc_auc": run.data.metrics.get("roc_auc", 0),
            "f1_score": run.data.metrics.get("f1_score", 0),
            "accuracy": run.data.metrics.get("accuracy", 0),
        }
    )

# Format and display as a structured DataFrame
df = pd.DataFrame(run_data)
print("All Runs (sorted by ROC AUC):")
print(df)

#Task 1.3: Register Best Model

# Isolate the optimal run configuration
best_run = runs[0]
best_run_id = best_run.info.run_id

print(f"Best model candidate: {best_run.info.run_name}")
print(f"ROC AUC: {best_run.data.metrics['roc_auc']:.4f}\n")

# Model signature coordinates
model_name = "PredictiveMaintenance"
model_uri = f"runs:/{best_run_id}/model"

# Register the model artifact into the central registry repository
registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

print(
    f"Registered model: {registered_model.name} (Version {registered_model.version})"
)



#Part 2: Stage Transitions
#Task 2.1: Add Model Documentation
version = registered_model.version
roc_auc = best_run.data.metrics["roc_auc"]
f1 = best_run.data.metrics["f1_score"]

# Generate technical description release log
description = (
    f"XGBoost model for predictive maintenance. "
    f"Trained on equipment sensor data (10,000 samples). "
    f"Performance: ROC AUC: {roc_auc:.4f}, F1 Score: {f1:.4f}. "
    f"Features: temperature, vibration, pressure, rpm, age_days."
)

# Push description to tracking registry instance
client.update_model_version(
    name=model_name, version=version, description=description
)

# Append tracking metadata tags
client.set_model_version_tag(model_name, version, "validation_status", "passed")
client.set_model_version_tag(model_name, version, "team", "data-science")
client.set_model_version_tag(model_name, version, "framework", "xgboost")

print(f"Documentation and structural tags successfully appended to Version {version}!")

#Task 2.2: Transition to Staging
# Transition model version phase gate
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Staging",
    archive_existing_versions=True,
)
print(f"Model version {version} moved to Staging stage!\n")

staging_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/Staging"
)
print("Staging model loaded successfully!")

#Task 2.3: Test in Staging
# Create high-stress simulation sample point
test_data = pd.DataFrame(
    {
        "temperature": [95.0],  # Outlier
        "vibration": [0.9],  # Outlier
        "pressure": [135.0],  # Outlier
        "rpm": [1500.0],
        "age_days": [320],  # Long cycle duration
    }
)

# Run inference through the localized pre-production instance reference
prediction = staging_model.predict(test_data)

print("Test Prediction (Staging Model Evaluation):")
print("Input Vector: High temperature, extreme vibration, aged mechanical asset.")
print(
    f"Prediction: {'FAILURE LIKELY' if prediction[0] == 1 else 'NORMAL OPERATION'}"
)
print(f"Raw Output Class Code: {prediction[0]}")

#Part 3: Production Inference
#Task 3.1: Promote to Production

from datetime import datetime

# Promote the model artifact configuration to Production status
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Production",
    archive_existing_versions=True,
)
print(f"🚀 Model version {version} is officially live in PRODUCTION!")

# Apply historical compliance audit deployment timestamp
current_date = datetime.now().strftime("%Y-%m-%d")
client.set_model_version_tag(
    model_name, version, "deployment_date", current_date
)

#Task 3.2: Build Production Inference Pipeline

def predict_equipment_failure(temperature, vibration, pressure, rpm, age_days):
    """Production wrapper function leveraging live Model Registry targets."""
    # Automated production reference loading archetype
    prod_model = mlflow.pyfunc.load_model(
        model_uri="models:/PredictiveMaintenance/Production"
    )

    # Reconstruct dictionary into scaled tabular schema structure format
    input_data = pd.DataFrame(
        [
            {
                "temperature": temperature,
                "vibration": vibration,
                "pressure": pressure,
                "rpm": rpm,
                "age_days": age_days,
            }
        ]
    )

    # Compute execution assessment vector arrays
    prediction = prod_model.predict(input_data)[0]

    return {
        "will_fail": bool(prediction),
        "recommendation": (
            "Schedule Emergency Maintenance Immediately!"
            if prediction
            else "Normal Baseline Fleet Operations Management."
        ),
    }


# Operational scenario array test metrics blocks
scenarios = [
    {
        "name": "Normal Operation Baseline",
        "temp": 70.0,
        "vib": 0.4,
        "press": 95.0,
        "rpm": 1500.0,
        "age": 100,
    },
    {
        "name": "Medium Thermal Warning Vector",
        "temp": 85.0,
        "vib": 0.6,
        "press": 110.0,
        "rpm": 1500.0,
        "age": 200,
    },
    {
        "name": "High Risk Critical Fault Alert",
        "temp": 95.0,
        "vib": 0.9,
        "press": 135.0,
        "rpm": 1500.0,
        "age": 320,
    },
]

print("Executing Production Pipeline Scenarios:\n")
for s in scenarios:
    result = predict_equipment_failure(
        s["temp"], s["vib"], s["press"], s["rpm"], s["age"]
    )
    print(f"Scenario: {s['name']:28}")
    print(f"Recommendation Result: {result['recommendation']}\n")

#Task 3.3: Test Model Rollback

# Simulate registering a secondary model option using our second-best run configuration
second_run = runs[1]
v2_uri = f"runs:/{second_run.info.run_id}/model"

v2_model = mlflow.register_model(model_uri=v2_uri, name=model_name)
print(
    f"Simulated release: Registered {v2_model.name} (Version {v2_model.version})"
)

# Rollback Sequence: Re-target version 1 and force its promotion back to Production
print("\n🚨 Initiating model rollback strategy protocol...")
client.transition_model_version_stage(
    name=model_name,
    version=1,  # Target the baseline stable model version
    stage="Production",
    archive_existing_versions=True,  # This auto-archives Version 2 out of production
)

print("✨ Rollback operation completed successfully!")
print("The production inference gateway has rolled back to Version 1.")