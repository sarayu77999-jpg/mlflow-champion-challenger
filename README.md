Champion–Challenger Model Comparison

1. Task and Models

Task: Binary sentiment classification of movie reviews using pretrained Hugging Face models.

Dataset: Movie review sentiment dataset containing 872 labelled evaluation examples.

Model A – Initial Champion:
"distilbert-base-uncased-finetuned-sst-2-english"

Model B – Challenger:
"textattack/bert-base-uncased-SST-2"

Both models were evaluated using the same 872 labelled movie reviews.

Tools Used:

- Python
- Hugging Face Transformers
- PyTorch
- MLflow
- Pandas
- Scikit-learn
- Matplotlib

2. How to Run

Step 1: Install dependencies

pip install -r requirements.txt

Step 2: Start the MLflow server

python -m mlflow server --host 127.0.0.1 --port 5000

Step 3: Evaluate both models

Open a second terminal and run:

python evaluate.py

Step 4: Register Champion and Challenger

python register_champion.py

Step 5: Predict on new data

python predict_from_mlflow.py

Open the MLflow dashboard:

http://127.0.0.1:5000

3. Champion Selection Rule

Model A is the initial Champion, and Model B is the Challenger.

The primary evaluation metric is F1-score.

Model B replaces Model A only if:

"F1_B - F1_A >= 0.01"

Otherwise, Model A remains the Champion.

Inference latency is recorded as a secondary metric. Lower latency is preferable. If the F1-score improvement does not meet the required threshold, Model A remains Champion.

This rule was defined before the final evaluation.

4. Results Table

Both models were evaluated on the same 872 examples.

Metric| Model A| Model B
Accuracy| 0.9106| 0.9234
Precision| 0.8978| 0.9181
Recall| 0.9300| 0.9346
F1-score| 0.9137| 0.9263
Average latency (ms/sample)| 21.64| 44.66

F1-score difference:

"0.9263 - 0.9137 = 0.0126"

Winning model: Model B

5. Champion Decision

Model B (textattack/bert-base-uncased-SST-2) was selected as the Champion because it achieved an F1-score of 0.9263, compared to Model A's F1-score of 0.9137. The improvement was 0.0126, exceeding the predefined minimum improvement of 0.01.

Model B also achieved better accuracy, precision and recall. Its accuracy was 0.9234, while Model A achieved 0.9106. This indicates that Model B performed better overall on the evaluation dataset.

However, Model A was faster during inference. Its average latency was approximately 21.64 milliseconds per example, whereas Model B required approximately 44.66 milliseconds per example. Therefore, Model A has an advantage when prediction speed is important.

Despite the higher latency, Model B was selected because the improvement in F1-score satisfied the predefined Champion selection rule. The decision prioritised classification performance over inference speed.

Both models were registered in MLflow. Model B was assigned the "champion" alias, and Model A was assigned the "challenger" alias.

6. Limitations and AI Assistance Declaration

Limitations

- The evaluation was performed on a limited dataset of 872 movie reviews.
- Results may vary when tested on different datasets or types of text.
- Both models were pretrained and were not fine-tuned specifically for this assignment.
- Inference latency may vary depending on hardware and system performance.
- The evaluation results may not represent performance on all real-world sentiment classification tasks.

AI Assistance Declaration

AI assistance was used for understanding the assignment, Python implementation, debugging and documentation. The team reviewed the implementation and evaluation results and is responsible for understanding the code and explaining the model comparison and selection process during the viva.
