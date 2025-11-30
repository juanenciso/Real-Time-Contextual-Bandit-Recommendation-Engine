import numpy as np
from typing import List, Dict, Tuple


class LinUCBBandit:
    """
    Simple LinUCB for K discrete actions.

    For each action a:
      A_a (d x d) and b_a (d x 1)

    Feature vector x is R^d (context).
    """

    def __init__(self, n_actions: int, dim: int, alpha: float = 1.0):
        self.n_actions = n_actions
        self.dim = dim
        self.alpha = alpha

        self.A = [np.eye(dim) for _ in range(n_actions)]
        self.b = [np.zeros((dim, 1)) for _ in range(n_actions)]

    def _theta(self, a_idx: int) -> np.ndarray:
        A_inv = np.linalg.inv(self.A[a_idx])
        return A_inv @ self.b[a_idx]  # (d x 1)

    def choose_action(self, x: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        x: (dim,) feature vector.
        Returns:
          index of action, ucb_score, list of scores for all actions
        """
        x = x.reshape(-1, 1)  # (d,1)
        scores = []
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mu = float(theta.T @ x)
            sigma = float(self.alpha * np.sqrt(x.T @ A_inv @ x))
            scores.append(mu + sigma)
        scores_arr = np.array(scores)
        a_star = int(np.argmax(scores_arr))
        return a_star, float(scores_arr[a_star]), scores_arr

    def update(self, a_idx: int, x: np.ndarray, reward: float) -> None:
        x = x.reshape(-1, 1)
        self.A[a_idx] += x @ x.T
        self.b[a_idx] += reward * x

    def get_params(self) -> Dict:
        return {
            "A": np.stack(self.A, axis=0),
            "b": np.stack(self.b, axis=0),
            "alpha": self.alpha,
        }

    @classmethod
    def from_params(cls, params: Dict) -> "LinUCBBandit":
        A = params["A"]
        b = params["b"]
        alpha = float(params["alpha"])

        n_actions, dim, _ = A.shape
        bandit = cls(n_actions=n_actions, dim=dim, alpha=alpha)
        bandit.A = [A[i] for i in range(n_actions)]
        bandit.b = [b[i] for i in range(n_actions)]
        return bandit

