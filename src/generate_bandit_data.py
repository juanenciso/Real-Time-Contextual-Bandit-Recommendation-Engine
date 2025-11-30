import numpy as np
import pandas as pd
from pathlib import Path


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

np.random.seed(42)


ACTIONS = [
    "show_bonus_offer",
    "suggest_new_app",
    "invite_friend",
    "deep_link_to_store",
]


def simulate_bandit_logs(
    n_users: int = 3000,
    sessions_per_user: int = 10,
    eps_logging: float = 0.3,
) -> pd.DataFrame:
    """
    Simulate contextual bandit logs for a rewarded app.

    Each row = one interaction:
      - user context
      - chosen action (by logging policy)
      - reward (1/0, e.g. completion of high-value action)
    """

    user_ids = np.arange(1, n_users + 1)

    # User-level latent preferences per action
    # shape: n_users x n_actions
    n_actions = len(ACTIONS)
    user_pref = np.random.normal(loc=0.0, scale=1.0, size=(n_users, n_actions))

    rows = []

    countries = ["AT", "DE", "ES", "IT"]
    devices = ["ios", "android"]
    segments = ["low", "mid", "high"]

    for u in user_ids:
        # assign user static attributes
        country = np.random.choice(countries)
        device = np.random.choice(devices)
        segment = np.random.choice(segments, p=[0.4, 0.4, 0.2])

        # base engagement profile
        base_engagement = {
            "low": (1, 30),
            "mid": (2, 60),
            "high": (3, 120),
        }[segment]

        for t in range(sessions_per_user):
            n_sessions = t + 1
            days_since_install = np.random.randint(0, 30)

            # context numeric features
            recent_engagement = base_engagement[0] + np.random.poisson(1)
            avg_session_length = base_engagement[1] + np.random.normal(0, 10)
            avg_session_length = max(avg_session_length, 5.0)

            # choose action with epsilon-greedy logging policy
            u_idx = u - 1
            prefs = user_pref[u_idx]

            if np.random.rand() < eps_logging:
                # random action
                a_idx = np.random.randint(0, n_actions)
            else:
                # greedy wrt latent preference (unknown to learner)
                a_idx = int(np.argmax(prefs))

            action = ACTIONS[a_idx]

            # reward probability as sigmoid of (preference + some context effect)
            context_bonus = 0.0
            if segment == "high":
                context_bonus += 0.3
            elif segment == "mid":
                context_bonus += 0.1

            if device == "ios":
                context_bonus += 0.05

            raw_score = prefs[a_idx] + 0.2 * np.log1p(recent_engagement) + context_bonus
            p_reward = 1 / (1 + np.exp(-raw_score))  # sigmoid

            reward = np.random.binomial(1, p_reward)

            rows.append(
                {
                    "user_id": u,
                    "country": country,
                    "device": device,
                    "segment": segment,
                    "n_sessions": n_sessions,
                    "days_since_install": days_since_install,
                    "recent_engagement": recent_engagement,
                    "avg_session_length": avg_session_length,
                    "action": action,
                    "reward": reward,
                    "p_reward_true": p_reward,
                }
            )

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    print("Simulating contextual bandit logs...")
    df = simulate_bandit_logs()
    out_path = DATA_DIR / "bandit_logs.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {len(df)} interactions to {out_path}")

