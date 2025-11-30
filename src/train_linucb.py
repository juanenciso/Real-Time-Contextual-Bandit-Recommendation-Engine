import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

from src.linucb_bandit import LinUCBBandit


DATA_DIR = Path("data")
MODEL_PATH = DATA_DIR / "linucb_model.npz"

ACTIONS = [
    "show_bonus_offer",
    "suggest_new_app",
    "invite_friend",
    "deep_link_to_store",
]


def build_feature_vector(row: pd.Series) -> np.ndarray:
    """
    Map row -> numeric feature vector for LinUCB.

    Includes:
      - one-hot for country
      - one-hot for device
      - one-hot for segment
      - scaled numeric features
    """

    countries = ["AT", "DE", "ES", "IT"]
    devices = ["ios", "android"]
    segments = ["low", "mid", "high"]

    feat = []

    # one-hot country
    for c in countries:
        feat.append(1.0 if row["country"] == c else 0.0)

    # one-hot device
    for d in devices:
        feat.append(1.0 if row["device"] == d else 0.0)

    # one-hot segment
    for s in segments:
        feat.append(1.0 if row["segment"] == s else 0.0)

    # numeric features (normalized roughly)
    feat.append(row["n_sessions"] / 20.0)
    feat.append(row["days_since_install"] / 30.0)
    feat.append(row["recent_engagement"] / 20.0)
    feat.append(row["avg_session_length"] / 200.0)

    return np.array(feat, dtype=np.float32)


def main(alpha: float = 1.0):
    df = pd.read_csv(DATA_DIR / "bandit_logs.csv")
    print(f"Loaded {len(df)} interactions from bandit_logs.csv")

    action_to_idx = {a: i for i, a in enumerate(ACTIONS)}

    # Build feature matrix dims
    x_example = build_feature_vector(df.iloc[0])
    dim = x_example.shape[0]

    bandit = LinUCBBandit(n_actions=len(ACTIONS), dim=dim, alpha=alpha)

    n = len(df)
    print(f"Feature dim = {dim}, actions = {len(ACTIONS)}")

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        x = build_feature_vector(row)
        a_idx = action_to_idx[row["action"]]
        r = float(row["reward"])

        # In offline training, we just update with the logged action
        bandit.update(a_idx, x, r)

        if i % 5000 == 0:
            print(f"  processed {i}/{n} interactions...")

    params = bandit.get_params()
    np.savez(
        MODEL_PATH,
        A=params["A"],
        b=params["b"],
        alpha=params["alpha"],
        actions=np.array(ACTIONS),
    )
    print(f"✅ Saved LinUCB model to {MODEL_PATH}")


if __name__ == "__main__":
    main(alpha=1.0)

