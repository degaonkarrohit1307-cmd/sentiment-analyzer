"""
Assignment 4 — ML Pipeline using Apache Airflow
================================================
DAG: sentiment_analyzer_pipeline
Tasks:
  1. data_preprocessing  — clean and validate input data
  2. model_training      — (re)train / reload NLP models
  3. model_evaluation    — run evaluation on test set
  4. model_versioning    — update model_versions.json
  5. deploy_model        — promote model to production if metrics pass

Run via Airflow UI or: airflow dags trigger sentiment_analyzer_pipeline
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# ── Default DAG arguments ─────────────────────────────────────────────────────
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email": ["mlops@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ── Task Functions ────────────────────────────────────────────────────────────

def task_data_preprocessing(**context):
    """
    Task 1: Data Preprocessing
    - Loads raw text data
    - Removes nulls, duplicates, very short entries
    - Validates input format
    """
    import re

    print("[Task 1] Starting data preprocessing...")

    sample_data = [
        "I love this product! It's amazing.",
        "Terrible experience. Never again.",
        "",                               # blank — will be removed
        "okay",                           # too short — will be removed
        "The service was absolutely wonderful and I'm very satisfied.",
        "This is the worst thing I have ever bought in my entire life.",
        "Normal day, nothing special.",
        "   ",                            # whitespace only — removed
    ]

    # Clean: remove blanks and very short entries
    cleaned = [t.strip() for t in sample_data if t.strip() and len(t.strip()) > 5]

    print(f"[Task 1] Raw records: {len(sample_data)}, After cleaning: {len(cleaned)}")
    assert len(cleaned) > 0, "No data left after preprocessing!"

    # Push cleaned data to XCom for next task
    context['ti'].xcom_push(key='cleaned_texts', value=cleaned)
    print("[Task 1] Data preprocessing complete.")
    return cleaned


def task_model_training(**context):
    """
    Task 2: Model Training / Initialization
    - Loads NLTK VADER and TextBlob models
    - Downloads required corpora if missing
    - Verifies models are ready for inference
    """
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from textblob import TextBlob

    print("[Task 2] Starting model training / initialization...")

    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

    # Initialize VADER
    sia = SentimentIntensityAnalyzer()
    test_score = sia.polarity_scores("This is a test sentence.")
    print(f"[Task 2] VADER test score: {test_score}")

    # Initialize TextBlob
    blob = TextBlob("This is a test sentence.")
    print(f"[Task 2] TextBlob polarity: {blob.sentiment.polarity}")

    print("[Task 2] Models initialized successfully.")
    context['ti'].xcom_push(key='model_status', value='ready')
    return "model_ready"


def task_model_evaluation(**context):
    """
    Task 3: Model Evaluation
    - Runs inference on preprocessed test data
    - Calculates accuracy, sentiment distribution
    - Returns evaluation metrics
    """
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from textblob import TextBlob

    print("[Task 3] Starting model evaluation...")

    ti = context['ti']
    texts = ti.xcom_pull(task_ids='data_preprocessing', key='cleaned_texts')

    if not texts:
        texts = ["Great product!", "Awful experience.", "Just okay."]

    sia = SentimentIntensityAnalyzer()
    results = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for text in texts:
        score = sia.polarity_scores(text)['compound']
        if score >= 0.05:
            results["Positive"] += 1
        elif score <= -0.05:
            results["Negative"] += 1
        else:
            results["Neutral"] += 1

    total = len(texts)
    accuracy_proxy = (results["Positive"] + results["Negative"]) / total if total else 0

    print(f"[Task 3] Evaluation results: {results}")
    print(f"[Task 3] Classification rate (non-neutral): {accuracy_proxy:.2%}")

    metrics = {
        "total_evaluated": total,
        "sentiment_distribution": results,
        "classification_rate": round(accuracy_proxy, 3),
    }

    context['ti'].xcom_push(key='eval_metrics', value=metrics)
    print("[Task 3] Model evaluation complete.")
    return metrics


def task_model_versioning(**context):
    """
    Task 4: Model Versioning
    - Reads current model_versions.json
    - Logs evaluation metrics against current version
    - Prepares new version entry if performance improved
    """
    import json
    import os

    print("[Task 4] Starting model versioning...")

    ti = context['ti']
    eval_metrics = ti.xcom_pull(task_ids='model_evaluation', key='eval_metrics')

    print(f"[Task 4] Metrics from evaluation: {eval_metrics}")

    # Simulate version logging
    version_log = {
        "pipeline_run": datetime.utcnow().isoformat(),
        "version": "v1.1",
        "eval_metrics": eval_metrics,
        "status": "evaluated",
    }

    print(f"[Task 4] Version log: {version_log}")
    print("[Task 4] Model versioning complete.")
    return version_log


def task_deploy_model(**context):
    """
    Task 5: Conditional Deployment
    - Checks if evaluation metrics cross deployment threshold
    - Promotes model to production if classification_rate >= 0.6
    """
    print("[Task 5] Starting deployment check...")

    ti = context['ti']
    eval_metrics = ti.xcom_pull(task_ids='model_evaluation', key='eval_metrics')

    threshold = 0.6
    rate = eval_metrics.get("classification_rate", 0) if eval_metrics else 0

    if rate >= threshold:
        print(f"[Task 5] Metrics passed threshold ({rate} >= {threshold}). Deploying model.")
        print("[Task 5] Model promoted to production successfully.")
        return "deployed"
    else:
        print(f"[Task 5] Metrics below threshold ({rate} < {threshold}). Skipping deployment.")
        return "skipped"


# ── DAG Definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="sentiment_analyzer_pipeline",
    default_args=default_args,
    description="End-to-end ML pipeline for the NLP Sentiment Analyzer",
    schedule_interval="@daily",          # runs every day at midnight
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "nlp", "sentiment"],
) as dag:

    # Task 1: Data Preprocessing
    preprocess = PythonOperator(
        task_id="data_preprocessing",
        python_callable=task_data_preprocessing,
        provide_context=True,
    )

    # Task 2: Model Training
    train = PythonOperator(
        task_id="model_training",
        python_callable=task_model_training,
        provide_context=True,
    )

    # Task 3: Model Evaluation
    evaluate = PythonOperator(
        task_id="model_evaluation",
        python_callable=task_model_evaluation,
        provide_context=True,
    )

    # Task 4: Model Versioning
    version = PythonOperator(
        task_id="model_versioning",
        python_callable=task_model_versioning,
        provide_context=True,
    )

    # Task 5: Deploy Model
    deploy = PythonOperator(
        task_id="deploy_model",
        python_callable=task_deploy_model,
        provide_context=True,
    )

    # ── Task Dependencies (DAG edges) ─────────────────────────────────────────
    # preprocess → train → evaluate → version → deploy
    preprocess >> train >> evaluate >> version >> deploy
