"""Evaluate both SST-2 models and log models usable by the project scripts."""

import argparse
import hashlib
import os
from contextlib import nullcontext
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "hf-sentiment-champion-challenger"
DATASET_ID = "stanfordnlp/sst2"
VALIDATION_FILE = "data/validation-00000-of-00001.parquet"
MODELS = (
    ("A", "distilbert-base-uncased-finetuned-sst-2-english"),
    ("B", "textattack/bert-base-uncased-SST-2"),
)
LABELS = ("NEGATIVE", "POSITIVE")
LABEL_IDS = {label: index for index, label in enumerate(LABELS)}


def load_evaluation_data(path):
    import pandas as pd

    if path.exists():
        data = pd.read_csv(path, dtype={"text": str}, keep_default_na=False)
    else:
        from huggingface_hub import hf_hub_download

        validation_path = hf_hub_download(DATASET_ID, VALIDATION_FILE, repo_type="dataset")
        data = pd.read_parquet(validation_path, columns=["sentence", "label"])
        data = data.rename(columns={"sentence": "text"})[["text", "label"]]
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        print(f"Created {path} from {DATASET_ID} validation.")

    if not {"text", "label"}.issubset(data.columns):
        raise ValueError(f"{path} must contain text and label columns.")
    if data["text"].str.strip().eq("").any():
        raise ValueError(f"{path} contains blank text.")
    label_text = data["label"].astype(str).str.strip().str.upper()
    labels = label_text.map({**LABEL_IDS, "0": 0, "1": 1})
    if labels.isna().any() or set(labels) != {0, 1}:
        raise ValueError(f"{path} must contain both classes (0/negative, 1/positive).")
    return data[["text"]].assign(label=labels.astype(int))


def load_classifier(source, batch_size):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)
    # TextAttack BERT uses generic LABEL_0/LABEL_1 for the SST-2 class IDs.
    if model.config.id2label == {0: "LABEL_0", 1: "LABEL_1"}:
        model.config.id2label = dict(enumerate(LABELS))
        model.config.label2id = LABEL_IDS.copy()
    if set(model.config.id2label.values()) != set(LABELS):
        raise ValueError(f"Unexpected sentiment labels for {source}: {model.config.id2label}")
    return pipeline(
        "text-classification", model=model, tokenizer=tokenizer,
        device="cpu", batch_size=batch_size,
    )


def predict_frame(classifier, frame):
    import pandas as pd

    rows = classifier(frame["text"].tolist(), truncation=True, max_length=512)
    return pd.DataFrame(
        ((row["label"], row["score"]) for row in rows),
        columns=["prediction", "confidence"],
    ).astype({"prediction": "object", "confidence": "float64"})


def save_mlflow_model(classifier, batch_size, example, predictions):
    import mlflow
    from mlflow.models import Model, infer_signature

    # Define the wrapper locally so MLflow serializes it with its inference code.
    class SentimentModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.classifier = load_classifier(context.artifacts["checkpoint"], batch_size)

        def predict(self, context, model_input, params=None):
            return predict_frame(self.classifier, model_input)

    with TemporaryDirectory() as temp:
        checkpoint = Path(temp) / "checkpoint"
        model_path = Path(temp) / "model"
        classifier.save_pretrained(checkpoint)
        mlflow.pyfunc.save_model(
            path=str(model_path),
            python_model=SentimentModel(),
            artifacts={"checkpoint": str(checkpoint)},
            signature=infer_signature(example, predictions),
            input_example=example,
            pip_requirements=[
                f"{package}=={version(package).split('+')[0]}"
                for package in ("mlflow", "cloudpickle", "pandas", "torch", "transformers")
            ],
            mlflow_model=Model(run_id=mlflow.active_run().info.run_id, artifact_path="model"),
        )
        # Keep a physical run artifact for runs:/<run_id>/model on MLflow 2 and 3.
        mlflow.log_artifacts(str(model_path), artifact_path="model")


def evaluate_model(classifier, data, output, model_letter, batch_size):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    inputs = data[["text"]]
    predict_frame(classifier, inputs.head(batch_size))
    started = perf_counter()
    predictions = predict_frame(classifier, inputs)
    elapsed = perf_counter() - started
    predicted = predictions["prediction"].map(LABEL_IDS)
    actual = data["label"]
    metrics = {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "inference_seconds": elapsed,
        "latency_ms_per_example": 1000 * elapsed / len(data),
    }
    output.mkdir(parents=True)
    data.join(predictions).to_csv(output / "predictions.csv", index=False)
    display = ConfusionMatrixDisplay.from_predictions(
        actual, predicted, labels=[0, 1], display_labels=LABELS, cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title(f"Model {model_letter} sentiment confusion matrix")
    display.figure_.savefig(output / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(display.figure_)
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "eval_inputs.csv")
    parser.add_argument("--tracking-uri", default=os.environ.get(
        "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
    ))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Evaluate up to four examples per class and skip MLflow.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    if not args.smoke_test:
        import mlflow

        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(EXPERIMENT)

    data = load_evaluation_data(args.data)
    if args.smoke_test:
        data = data.groupby("label", sort=False).head(4)
    data = data.reset_index(drop=True)
    evaluation_csv = data.to_csv(index=False)
    dataset_hash = hashlib.sha256(evaluation_csv.encode("utf-8")).hexdigest()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evaluation_id = f"{timestamp}_{uuid4().hex[:8]}"
    folder = "smoke_test" if args.smoke_test else "evaluation"
    output = ROOT / "outputs" / folder / evaluation_id
    output.mkdir(parents=True)
    snapshot = output / "eval_inputs.csv"
    snapshot.write_bytes(evaluation_csv.encode("utf-8"))
    print(f"Evaluating {len(data)} rows per model.", flush=True)

    for model_letter, model_id in MODELS:
        run_name = f"{model_letter}_sentiment_{evaluation_id}"
        print(f"Loading {model_id} ...", flush=True)
        run_context = (
            nullcontext() if args.smoke_test else mlflow.start_run(run_name=run_name)
        )
        with run_context as run:
            classifier = load_classifier(model_id, args.batch_size)
            model_output = output / run_name
            metrics, predictions = evaluate_model(
                classifier, data, model_output, model_letter, args.batch_size
            )
            if run is not None:
                mlflow.log_params({
                    "hf_model_id": model_id,
                    "evaluation_rows": len(data),
                    "evaluation_sha256": dataset_hash,
                    "batch_size": args.batch_size,
                    "device": "cpu",
                    "f1_average": "binary",
                    "positive_label": "POSITIVE",
                })
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(str(snapshot), artifact_path="evaluation")
                mlflow.log_artifacts(str(model_output), artifact_path="evaluation")
                save_mlflow_model(
                    classifier, args.batch_size, data[["text"]].head(2), predictions.head(2)
                )
                print(f"Logged {run_name} as {run.info.run_id}")
            print(f"{run_name}  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            del classifier

    print(f"Evaluation files saved to {output}")


if __name__ == "__main__":
    main()
