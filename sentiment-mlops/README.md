# 💬 SentimentLab — NLP Sentiment Analyzer (MLOps Edition)

A production-ready NLP Sentiment Analyzer built with Flask, NLTK VADER, and TextBlob — integrated with a complete MLOps pipeline.

## 🔍 Project Overview
This app analyzes sentiment of tweets/social media comments and classifies them as Positive, Negative, or Neutral using NLTK VADER + TextBlob.

## 📦 Dataset
- **Built-in**: 25 sample social media tweets covering diverse sentiments
- **Custom**: Users can paste any text/comments for live analysis

## 🧰 Tech Stack
| Layer | Technology |
|---|---|
| Backend | Python + Flask |
| NLP Engine | NLTK VADER + TextBlob |
| Frontend | HTML + CSS + Chart.js |
| Containerization | Docker |
| Orchestration | Kubernetes (Minikube) |
| CI/CD | GitHub Actions |
| Pipeline | Apache Airflow |
| Monitoring | Prometheus + Grafana |
| Logging | EFK Stack (Elasticsearch + Fluentd + Kibana) |

## 🚀 Features
- Sentiment classification (Positive / Negative / Neutral)
- Emotion detection (Joy, Optimism, Anger, Sadness, Neutral)
- Keyword frequency analysis
- Sentiment timeline chart
- A/B testing between two model versions
- Model versioning with performance tracking
- Auto-scaling via Kubernetes HPA
- Real-time monitoring via Prometheus metrics

## ▶️ How to Run Locally
```bash
pip install -r sentiment-analyzer/requirements.txt
python sentiment-analyzer/app.py
# Visit http://localhost:5000
```

## 🐳 Run with Docker
```bash
docker build -t sentiment-analyzer .
docker run -p 5000:5000 sentiment-analyzer
```

## ☸️ Deploy on Kubernetes
```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml
```

## 🔁 CI/CD
GitHub Actions automatically triggers on every push to `main`:
- Installs dependencies
- Runs tests
- Builds Docker image
- Deploys to test environment

## 📊 Monitoring
- Prometheus metrics at `/metrics`
- Health check at `/health`
- A/B test results at `/ab-test`

## 👥 Collaboration
- Fork this repository
- Create a feature branch
- Submit a Pull Request for review

## 📁 Project Structure
```
sentiment-mlops/
├── sentiment-analyzer/       # Core Flask app
│   ├── app.py                # Main application
│   ├── requirements.txt      # Dependencies
│   └── templates/            # HTML frontend
├── airflow/dags/             # Assignment 4 - ML Pipeline
├── kubernetes/               # Assignment 5, 8 - K8s manifests
├── monitoring/               # Assignment 9 - Prometheus config
├── .github/workflows/        # Assignment 6 - CI/CD
├── Dockerfile                # Assignment 3 - Containerization
├── model_versions.json       # Assignment 2 - Model versioning
└── README.md
```
