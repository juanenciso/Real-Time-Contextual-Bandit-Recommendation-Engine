
🎯 Real-Time Contextual Bandit Recommendation Engine

A production-grade reinforcement learning engine for real-time personalized recommendations in rewarded mobile apps.

🚀 Overview

This project implements a real-time Contextual Bandit recommender system (LinUCB) that predicts the next best action to maximize:

Engagement

Retention

Session depth

Revenue per user

Recommended actions include:

🎁 Bonus offer

⭐ Invite a friend

📱 Suggest new app

🔗 Deep link to store

🧠 Key Features

Synthetic dataset generator (user → context → reward)

LinUCB contextual bandit implementation

Training pipeline with reproducible results

Real-time scoring API (FastAPI)

Online-learning architecture

Persistent model storage (npz)

Industry-level project structure

🏗 Architecture
📌 High-Level Flow
User Context
      ↓
Feature Vector
      ↓
LinUCB Model
      ↓
UCB Scores
      ↓
Best Action Selected

📂 Project Structure
realtime-reco-bandit-engine/
│
├── src/
│   ├── generate_bandit_data.py    # Generates synthetic training data
│   ├── train_linucb.py            # Trains the LinUCB model
│   ├── linucb_bandit.py           # LinUCB implementation (core logic)
│
├── api_recommender.py             # FastAPI microservice for real-time recommendations
├── requirements.txt
└── README.md

💾 Installation
git clone https://github.com/juanenciso/Real-Time-Contextual-Bandit-Recommendation-Engine.git
cd Real-Time-Contextual-Bandit-Recommendation-Engine

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

🧪 Step 1 — Generate Synthetic Data
python src/generate_bandit_data.py

🏋️ Step 2 — Train the LinUCB Model
python src/train_linucb.py


This creates:

data/linucb_model.npz

⚡ Step 3 — Run the API Server
uvicorn api_recommender:app --reload --port 8020

📡 Step 4 — Example API Request
curl -X POST "http://127.0.0.1:8020/recommend_action" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u123",
    "country": "AT",
    "device": "ios",
    "segment": "high",
    "n_sessions": 10,
    "days_since_install": 5,
    "recent_engagement": 12,
    "avg_session_length": 180.0
  }'


Example response:

{
  "user_id": "u123",
  "recommended_action": "show_bonus_offer",
  "ucb_score": 1.03,
  "scores": {...}
}

📊 Why This Project Stands Out

This project demonstrates real production-level ML engineering skills, including:

Reinforcement learning (Contextual Bandits)

Online inference

Vectorized model serving

Feature engineering

Experiment reproducibility

Clean architecture

Deployable microservice (FastAPI + Uvicorn)

👨‍💻 Author

Juan Sebastián Enciso García, PhD
Data Scientist • ML Engineer

