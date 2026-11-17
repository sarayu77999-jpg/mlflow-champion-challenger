
# Champion-Challenger Model Comparison

## 1. Task

Binary sentiment classification of movie reviews using
pretrained Hugging Face models.

### Model A: Current Champion
distilbert-base-uncased-finetuned-sst-2-english

### Model B: Challenger
textattack/bert-base-uncased-SST-2

## 2. Tools

- Python
- Hugging Face Transformers
- PyTorch
- MLflow
- Pandas
- Scikit-learn
- Matplotlib

## 3. Dataset

Both models are evaluated on the same labelled movie
reviews. The evaluation dataset is stored in:

data/eval_inputs.csv

The final Champion is tested on 10 new reviews stored in:

data/new_data.csv

## 4. Champion Decision Rule

Primary metric: F1-score.

Higher F1-score is better.

Model A is the initial Champion.

Model B replaces Model A only if:

F1_B - F1_A >= 0.01

Otherwise, Model A remains Champion.

Inference latency is recorded as a secondary metric.
Lower latency is preferable. When the F1 improvement
is below the required margin, Model A remains Champion,
and latency is discussed as a trade-off.

This rule is defined before the final evaluation.

## 5. How to Run

Install dependencies:

pip install -r requirements.txt

Start the MLflow server:

python -m mlflow server --host 127.0.0.1 --port 5000

In a separate terminal, execute:

python evaluate.py
python register_champion.py
python predict_from_mlflow.py

Open MLflow at:

http://127.0.0.1:5000

## 6. MLflow Configuration

Experiment:
hf-sentiment-champion-challenger

Registered model:
sentiment-model

Registry aliases:
champion
challenger

Evaluation metric:
f1

Logged model path/name:
model

Model input:
Pandas DataFrame containing a text column.

## 7. Results

To be completed after evaluation.

| Metric | Model A | Model B |
|--------|---------|---------|
| Accuracy | TBD | TBD |
| Precision | TBD | TBD |
| Recall | TBD | TBD |
| F1-score | TBD | TBD |
| Average latency (ms) | TBD | TBD |
| Total inference time (s) | TBD | TBD |

## 8. Champion Decision

To be completed after running both models.

Explain:
- Which model won?
- What was the F1-score difference?
- Was the difference meaningful?
- Which model was faster?
- What were the trade-offs?

## 9. Prediction on New Data

predict_from_mlflow.py loads the model using:

models:/sentiment-model@champion

It does not directly invoke the Hugging Face pipeline.

The script saves predictions to:

outputs/new_data_predictions.csv

## 10. Limitations

The evaluation uses a relatively small subset of
movie reviews. Results may vary on other datasets.

The pretrained models are not fine-tuned for this
assignment.

## 11. AI Assistance Declaration

AI assistance was used to help develop and explain
the Python implementation and documentation.

The code and results must be reviewed and understood
before submission.
