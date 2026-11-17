
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
MODEL_NAME = "sentiment-model"

PRIMARY_METRIC = "f1"
MARGIN = 0.01

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise RuntimeError(
        "MLflow experiment not found. Run evaluate.py first."
    )

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    output_format="list"
)


def find_run(prefix):
    matches = [
        run for run in runs
        if run.info.run_name.startswith(prefix)
        and run.info.status == "FINISHED"
        and PRIMARY_METRIC in run.data.metrics
    ]

    if not matches:
        raise RuntimeError(
            f"No completed evaluation run found for {prefix}"
        )

    return max(
        matches,
        key=lambda run: run.info.start_time
    )


run_a = find_run("A_sentiment_")
run_b = find_run("B_sentiment_")

f1_a = run_a.data.metrics[PRIMARY_METRIC]
f1_b = run_b.data.metrics[PRIMARY_METRIC]

delta = f1_b - f1_a

print("MODEL COMPARISON")
print("----------------")
print(f"Model A F1: {f1_a:.4f}")
print(f"Model B F1: {f1_b:.4f}")
print(f"F1 improvement (B - A): {delta:.4f}")

# Predefined decision rule:
# B replaces A only if F1 improves by at least 0.01.

if delta >= MARGIN:
    winner = "B"
    champion_run = run_b
    challenger_run = run_a
else:
    winner = "A"
    champion_run = run_a
    challenger_run = run_b

print(f"\nWinner: Model {winner}")

# Register the two logged models.
# Person 1 must log each model under the path "model".

def register_run(run):
    model_uri = f"runs:/{run.info.run_id}/model"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    return registered.version


champion_version = register_run(champion_run)
challenger_version = register_run(challenger_run)

# Assign MLflow registry aliases.

client.set_registered_model_alias(
    MODEL_NAME,
    "champion",
    champion_version
)

client.set_registered_model_alias(
    MODEL_NAME,
    "challenger",
    challenger_version
)

# Log the comparison in a summary run.

with mlflow.start_run(run_name="compare_sentiment"):
    mlflow.log_param("primary_metric", PRIMARY_METRIC)
    mlflow.log_param("margin", MARGIN)
    mlflow.log_param("model_a_run_id", run_a.info.run_id)
    mlflow.log_param("model_b_run_id", run_b.info.run_id)

    mlflow.log_metric("delta_primary_metric", delta)
    mlflow.set_tag("winner", winner)

print("\nRegistration completed successfully!")
print("Champion version:", champion_version)
print("Challenger version:", challenger_version)
