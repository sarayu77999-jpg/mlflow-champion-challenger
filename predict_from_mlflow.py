
import os
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "sentiment-model"

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()

# Find the model currently assigned the Champion alias.

version = client.get_model_version_by_alias(
    MODEL_NAME,
    "champion"
)

print("CHAMPION MODEL")
print("--------------")
print("Registered model:", MODEL_NAME)
print("Model version:", version.version)

# Retrieve the original Hugging Face model ID.

run = client.get_run(version.run_id)

hf_model_id = run.data.params.get(
    "hf_model_id",
    "Unknown"
)

print("Hugging Face model:", hf_model_id)

# Load the model using MLflow.
# No direct Hugging Face pipeline calls are used here.

model_uri = f"models:/{MODEL_NAME}@champion"

model = mlflow.pyfunc.load_model(model_uri)

# Read new input data.

data = pd.read_csv("data/new_data.csv")

if "text" not in data.columns:
    raise ValueError(
        "new_data.csv must contain a 'text' column."
    )

# Run predictions through the MLflow model.

predictions = model.predict(data[["text"]])

# Convert predictions into a DataFrame.

if isinstance(predictions, pd.DataFrame):
    results = pd.concat(
        [
            data.reset_index(drop=True),
            predictions.reset_index(drop=True)
        ],
        axis=1
    )
else:
    results = data.copy()
    results["prediction"] = predictions

print("\nNEW DATA PREDICTIONS")
print("--------------------")
print(results.to_string(index=False))

# Save the results.

os.makedirs("outputs", exist_ok=True)

output_path = "outputs/new_data_predictions.csv"

results.to_csv(output_path, index=False)

print(f"\nPredictions saved to {output_path}")
