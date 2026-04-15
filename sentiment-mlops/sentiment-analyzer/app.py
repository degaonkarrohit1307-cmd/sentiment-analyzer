"""
NLP Sentiment Analyzer - Flask Backend (MLOps Edition)
Stack  : Python + Flask + NLTK + TextBlob
MLOps  : Model Versioning + A/B Testing + Prometheus Metrics + Logging
"""

import os
import re
import json
import time
import logging
import random
from collections import Counter
from datetime import datetime
from flask import Flask, request, jsonify, render_template

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

# ── Logging Setup (Assignment 9) ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log"),
    ],
)
logger = logging.getLogger("sentiment-analyzer")

# ── Download required NLTK data ───────────────────────────────────────────────
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# ── Prometheus Metrics (Assignment 9) ────────────────────────────────────────
METRICS = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_failed": 0,
    "texts_analyzed_total": 0,
    "ab_test_v1_requests": 0,
    "ab_test_v2_requests": 0,
    "response_times": [],
    "sentiment_counts": {"Positive": 0, "Negative": 0, "Neutral": 0},
}

# ── Model Version Config (Assignment 2) ──────────────────────────────────────
def load_model_versions():
    try:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "model_versions.json")
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {"current_production_version": "v1.1", "ab_test_version": "v2.0", "versions": []}

MODEL_CONFIG = load_model_versions()

VERSION_THRESHOLDS = {
    "v1.1": {"positive": 0.05, "negative": -0.05, "subjectivity": 0.6, "emotion_high": 0.5, "emotion_low": -0.5},
    "v2.0": {"positive": 0.08, "negative": -0.08, "subjectivity": 0.55, "emotion_high": 0.45, "emotion_low": -0.45},
}

# ── Sample Data ───────────────────────────────────────────────────────────────
SAMPLE_TWEETS = [
    "Absolutely love the new iPhone update! Best thing Apple has done in years",
    "This product is terrible. Worst purchase I've ever made. Complete waste of money.",
    "Just had the most amazing coffee at the local cafe. Starting the day right",
    "Traffic is so bad today. Why does this always happen on Monday mornings",
    "The new Marvel movie was okay. Not bad, not great, just average I guess.",
    "Can't believe how bad the customer service was. Never going back to that store!",
    "Had such a wonderful birthday party with family. Life is beautiful",
    "The weather today is absolutely awful. Rain all day long, so depressing.",
    "Neutral thoughts about the new policy. It has both pros and cons.",
    "So happy to announce I got the job offer! Dreams do come true",
    "This app keeps crashing. The developers really need to fix these bugs.",
    "Feeling pretty indifferent about the election results. Politics as usual.",
    "The food at the restaurant was delicious! Highly recommend to everyone.",
    "Lost my wallet today. This is the worst day ever",
    "The new season of that show is just fine. Nothing special about it.",
    "I am beyond happy with my new car. Best investment I've ever made!",
    "Another boring meeting that could have been an email. Classic corporate life.",
    "The concert last night was absolutely epic! Best night of my life",
    "Terrible flight experience. Delayed 4 hours and lost my luggage. Never again.",
    "Pretty standard day at work. Nothing exciting or bad, just normal.",
    "This gym is amazing! Lost 10 kg in 2 months thanks to their trainers",
    "Online shopping experience was horrible. Wrong item delivered twice!",
    "The sunset today was breathtaking. Nature never stops amazing me",
    "My laptop broke right before a big presentation. Could this get any worse?",
    "Meh, the book was neither good nor bad. Just average writing really.",
]

# ── NLP Helpers ───────────────────────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def get_sentiment_label(compound, thresholds):
    if compound >= thresholds["positive"]:
        return "Positive"
    elif compound <= thresholds["negative"]:
        return "Negative"
    else:
        return "Neutral"

def get_emotion(compound, subjectivity, thresholds):
    if compound >= thresholds["emotion_high"] and subjectivity > thresholds["subjectivity"]:
        return "Joy"
    elif compound >= thresholds["positive"]:
        return "Optimism"
    elif compound <= thresholds["emotion_low"] and subjectivity > thresholds["subjectivity"]:
        return "Anger"
    elif compound <= thresholds["negative"]:
        return "Sadness"
    else:
        return "Neutral"

def extract_keywords(texts):
    stop_words = set(stopwords.words('english'))
    all_words = []
    for text in texts:
        cleaned = clean_text(text.lower())
        try:
            tokens = word_tokenize(cleaned)
        except Exception:
            tokens = cleaned.split()
        words = [w for w in tokens if w.isalpha() and w not in stop_words and len(w) > 3]
        all_words.extend(words)
    word_freq = Counter(all_words)
    return [{"word": w, "count": c} for w, c in word_freq.most_common(15)]

def analyze_texts(texts, version="v1.1"):
    thresholds = VERSION_THRESHOLDS.get(version, VERSION_THRESHOLDS["v1.1"])
    sia = SentimentIntensityAnalyzer()
    results = []
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    emotion_counts = Counter()
    compound_scores = []
    subjectivity_scores = []

    for text in texts:
        if not text.strip():
            continue
        cleaned = clean_text(text)
        vader_scores = sia.polarity_scores(cleaned)
        blob = TextBlob(cleaned)
        compound = vader_scores['compound']
        subjectivity = blob.sentiment.subjectivity
        sentiment = get_sentiment_label(compound, thresholds)
        emotion = get_emotion(compound, subjectivity, thresholds)
        sentiment_counts[sentiment] += 1
        emotion_counts[emotion] += 1
        compound_scores.append(compound)
        subjectivity_scores.append(subjectivity)
        results.append({
            "text": text[:120] + "..." if len(text) > 120 else text,
            "sentiment": sentiment,
            "emotion": emotion,
            "compound": round(compound, 3),
            "positive": round(vader_scores['pos'], 3),
            "negative": round(vader_scores['neg'], 3),
            "neutral": round(vader_scores['neu'], 3),
            "subjectivity": round(subjectivity, 3),
            "polarity": round(blob.sentiment.polarity, 3),
        })

    total = len(results)
    avg_compound = round(sum(compound_scores) / total, 3) if total else 0
    avg_subjectivity = round(sum(subjectivity_scores) / total, 3) if total else 0
    overall_sentiment = get_sentiment_label(avg_compound, thresholds)
    keywords = extract_keywords(texts)

    for s, c in sentiment_counts.items():
        METRICS["sentiment_counts"][s] += c
    METRICS["texts_analyzed_total"] += total

    chunk_size = max(1, total // 8)
    timeline = []
    for i in range(0, total, chunk_size):
        chunk = compound_scores[i:i + chunk_size]
        if chunk:
            timeline.append({"label": f"#{i+1}-{i+len(chunk)}", "score": round(sum(chunk) / len(chunk), 3)})

    return {
        "total": total,
        "model_version": version,
        "overall_sentiment": overall_sentiment,
        "avg_compound": avg_compound,
        "avg_subjectivity": avg_subjectivity,
        "sentiment_counts": sentiment_counts,
        "emotion_counts": dict(emotion_counts),
        "results": results,
        "keywords": keywords,
        "timeline": timeline,
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    start_time = time.time()
    METRICS["requests_total"] += 1
    logger.info("POST /analyze received")
    try:
        body = request.get_json()
        if not body:
            METRICS["requests_failed"] += 1
            return jsonify({"success": False, "error": "Request body missing."}), 400
        raw = body.get("text", "").strip()
        if not raw:
            METRICS["requests_failed"] += 1
            return jsonify({"success": False, "error": "No text provided."}), 400
        texts = [t.strip() for t in raw.split("\n") if t.strip()]
        if not texts:
            METRICS["requests_failed"] += 1
            return jsonify({"success": False, "error": "No valid lines found."}), 400
        version = MODEL_CONFIG.get("current_production_version", "v1.1")
        data = analyze_texts(texts, version=version)
        elapsed = round((time.time() - start_time) * 1000, 2)
        METRICS["response_times"] = (METRICS["response_times"] + [elapsed])[-100:]
        METRICS["requests_success"] += 1
        logger.info(f"POST /analyze OK — {len(texts)} texts in {elapsed}ms using {version}")
        return jsonify({"success": True, "data": data})
    except Exception as e:
        METRICS["requests_failed"] += 1
        logger.error(f"POST /analyze ERROR: {str(e)}")
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500

@app.route("/ab-test", methods=["POST"])
def ab_test():
    """Assignment 7 — A/B Testing: routes traffic between v1.1 and v2.0."""
    logger.info("POST /ab-test received")
    try:
        body = request.get_json()
        if not body:
            return jsonify({"success": False, "error": "Request body missing."}), 400
        raw = body.get("text", "").strip()
        if not raw:
            return jsonify({"success": False, "error": "No text provided."}), 400
        texts = [t.strip() for t in raw.split("\n") if t.strip()]
        if not texts:
            return jsonify({"success": False, "error": "No valid lines found."}), 400

        assigned = random.choice(["v1.1", "v2.0"])
        if assigned == "v1.1":
            METRICS["ab_test_v1_requests"] += 1
        else:
            METRICS["ab_test_v2_requests"] += 1

        result_v1 = analyze_texts(texts, version="v1.1")
        result_v2 = analyze_texts(texts, version="v2.0")
        logger.info(f"POST /ab-test assigned={assigned}, {len(texts)} texts")

        return jsonify({
            "success": True,
            "assigned_version": assigned,
            "model_a": {"version": "v1.1", "label": "Production (v1.1)", "data": result_v1},
            "model_b": {"version": "v2.0", "label": "Candidate (v2.0)", "data": result_v2},
            "traffic_stats": {
                "v1_requests": METRICS["ab_test_v1_requests"],
                "v2_requests": METRICS["ab_test_v2_requests"],
            }
        })
    except Exception as e:
        logger.error(f"POST /ab-test ERROR: {str(e)}")
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500

@app.route("/versions", methods=["GET"])
def versions():
    """Assignment 2 — Returns all tracked model versions."""
    logger.info("GET /versions requested")
    return jsonify({"success": True, "data": MODEL_CONFIG})

@app.route("/metrics", methods=["GET"])
def metrics():
    """Assignment 9 — Prometheus-style metrics endpoint."""
    avg_response = 0
    if METRICS["response_times"]:
        avg_response = round(sum(METRICS["response_times"]) / len(METRICS["response_times"]), 2)
    payload = {
        "app": "sentiment-analyzer",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "requests": {
            "total": METRICS["requests_total"],
            "success": METRICS["requests_success"],
            "failed": METRICS["requests_failed"],
        },
        "texts_analyzed_total": METRICS["texts_analyzed_total"],
        "avg_response_time_ms": avg_response,
        "sentiment_distribution": METRICS["sentiment_counts"],
        "ab_test_traffic": {
            "v1_1_requests": METRICS["ab_test_v1_requests"],
            "v2_0_requests": METRICS["ab_test_v2_requests"],
        }
    }
    logger.info("GET /metrics scraped")
    return jsonify(payload)

@app.route("/sample", methods=["GET"])
def sample():
    return jsonify({"success": True, "data": "\n".join(SAMPLE_TWEETS)})

@app.route("/health", methods=["GET"])
def health():
    """Assignment 9 — Health check for Kubernetes liveness/readiness probes."""
    version = MODEL_CONFIG.get("current_production_version", "v1.1")
    logger.info("GET /health checked")
    return jsonify({
        "status": "running",
        "engine": "NLTK VADER + TextBlob",
        "model_version": version,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

if __name__ == "__main__":
    print("\n[OK] NLP Sentiment Analyzer (MLOps Edition) at http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
