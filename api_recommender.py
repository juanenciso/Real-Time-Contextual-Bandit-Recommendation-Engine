from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.linucb_bandit import LinUCBBandit
from src.train_linucb import build_feature_vector, ACTIONS, DATA_DIR


MODEL_PATH = DATA_DIR / "linucb_model.npz"

app = FastAPI(title="Real-Time Recommendation Engine (LinUCB)")


class UserContext(BaseModel):
    user_id: str
    country: str       # "AT", "DE", "ES", "IT"
    device: str        # "ios", "android"
    segment: str       # "low", "mid", "high"
    n_sessions: int
    days_since_install: int
    recent_engagement: int
    avg_session_length: float


# Load model at startup
print("Loading LinUCB model from disk...")
npz = np.load(MODEL_PATH, allow_pickle=True)
A = npz["A"]
b = npz["b"]
alpha = float(npz["alpha"])
actions = [str(a) for a in npz["actions"]]

bandit = LinUCBBandit.from_params({"A": A, "b": b, "alpha": alpha})
print(f"Model loaded. actions={actions}, dim={A.shape[1]}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "actions": actions,
        "dim": int(A.shape[1]),
    }


@app.post("/recommend_action")
def recommend_action(ctx: UserContext):
    # Build synthetic row to reuse build_feature_vector
    row = pd.Series(
        {
            "country": ctx.country,
            "device": ctx.device,
            "segment": ctx.segment,
            "n_sessions": ctx.n_sessions,
            "days_since_install": ctx.days_since_install,
            "recent_engagement": ctx.recent_engagement,
            "avg_session_length": ctx.avg_session_length,
        }
    )

    x = build_feature_vector(row)
    a_idx, ucb_score, scores_arr = bandit.choose_action(x)

    scores_dict = {
        actions[i]: float(scores_arr[i])
        for i in range(len(actions))
    }

    return {
        "user_id": ctx.user_id,
        "recommended_action": actions[a_idx],
        "ucb_score": ucb_score,
        "scores": scores_dict,
        "alpha": alpha,
    }

