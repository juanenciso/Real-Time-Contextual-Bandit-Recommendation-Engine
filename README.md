
# 🎯 Real-Time Contextual Bandit Recommendation Engine  
*A production-ready reinforcement learning engine for real-time personalized recommendations in rewarded mobile apps.*

---

## 🚀 Overview  
This project implements a **real-time Contextual Bandit recommender system (LinUCB)** optimized for rewarded apps.  
It selects the **next best action** to increase engagement, retention, and monetization.

Supported high-value actions:
- 🎁 Show bonus offer  
- ⭐ Invite friend  
- 📱 Suggest new app to explore  
- 🔗 Deep link to store  

The engine uses:
✔️ User context  
✔️ Online learning  
✔️ Explore–exploit optimization  
✔️ FastAPI microservice for real-time inference  

---

## 🧠 Key Features  
- **Synthetic user–action dataset generator**  
- **LinUCB contextual bandit implementation**  
- **Training pipeline** (learns action weights + confidence bounds)  
- **FastAPI scoring API**  
- **Persistent model storage** (`npz`)  
- **Industry-level project structure**

---

## 🏗 Architecture  


A Reinforcement Learning system for real-time personalized recommendations in rewarded mobile apps.

🚀 Overview

This project implements a real-time recommendation engine based on Contextual Bandits (LinUCB) to recommend high-value actions such as:

🎁 Show bonus offer

⭐ Invite friend

📱 Suggest new app to explore

🔗 Deep link to store

The engine uses user context + exploration/exploitation to pick the best next action in real time, backed by an online-learning model.

This repository includes:

✅ Data simulation
✅ Training of a LinUCB contextual bandit
✅ Action scoring API using FastAPI
✅ Real-time recommendation endpoint
✅ Reproducible environment + clean project structure

🧠 Architecture

realtime-reco-bandit-engine/
│
├── src/
│ ├── generate_bandit_data.py # Generates synthetic interaction data
│ ├── train_linucb.py # Trains LinUCB model
│ ├── linucb_bandit.py # Bandit algorithm implementation
│
├── api_recommender.py # FastAPI microservice (real-time recommendations)
├── requirements.txt
├── README.md
└── .gitignore

User Context → Feature Vector → LinUCB Model → UCB Scores → Selected Action

A high-level view:

src/generate_bandit_data.py     → Creates training data
src/linucb_bandit.py            → LinUCB implementation
src/train_linucb.py             → Trains the bandit & saves model
api_recommender.py              → FastAPI real-time scoring API
data/                           → Saved bandit model (linucb_model.npz)

📦 Installation

1. Clone the repository

git clone git@github.com:juanenciso/Real-Time-Contextual-Bandit-Recommendation-Engine.git
cd Real-Time-Contextual-Bandit-Recommendation-Engine

2. Create and activate the virtual environment

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

🛠 Training the LinUCB Bandit

Generate simulated bandit data

python src/generate_bandit_data.py

This produces:

data/bandit_simulated.csv

Train the LinUCB model

python src/train_linucb.py

This saves the trained model:

data/linucb_model.npz

⚡ Run the API Server

uvicorn api_recommender:app --reload --port 8020

Server runs at:

👉 http://127.0.0.1:8020

🔍 API Endpoints

Health check

GET /health

Response:

{
  "status": "ok",
  "model_loaded": true,
  "n_actions": 4,
  "alpha": 1.0
}

🎯 Real-Time Recommendation Endpoint

POST /recommend_action

Example request:

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
  "ucb_score": 1.0327,
  "scores": {
    "show_bonus_offer": 1.0327,
    "suggest_new_app": 0.9185,
    "invite_friend": 0.9970,
    "deep_link_to_store": 1.0113
  },
  "alpha": 1.0
}

🧮 Model: LinUCB Explained

The LinUCB algorithm balances:

Exploration: testing new actions

Exploitation: choosing best known action

Optimization principle:

UCB = expected_reward + α * uncertainty

Where:

expected_reward = θᵀx

uncertainty = sqrt(xᵀ A⁻¹ x)

This allows the model to adapt in real time as new users interact with the system.


