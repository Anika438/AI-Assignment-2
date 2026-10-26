"""Double Linear Q-learning agent for ambulance redeployment."""

import math
import random


class LinearQAgent:
    """Learn action values using Double Linear Q-learning with TD clipping and finite bounds.

    Maintains two estimator vectors (weights_a and weights_b) to eliminate maximization bias.
    """

    def __init__(
        self,
        state_size: int,
        num_actions: int,
        learning_rate: float = 0.005,
        discount_factor: float = 0.95,
        epsilon: float = 0.35,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.94,
        max_td_error: float = 5.0,
        max_weight: float = 20.0,
        seed: int | None = None,
    ) -> None:
        if state_size <= 0 or num_actions <= 0:
            raise ValueError("State size and action count must be positive")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if not 0 <= discount_factor <= 1:
            raise ValueError("Discount factor must be between 0 and 1")
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be between 0 and 1")

        self.state_size = state_size
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_td_error = max_td_error
        self.max_weight = max_weight
        self.rng = random.Random(seed)

        # Double Q-learning: two independent sets of weights
        self.weights_a: list[list[float]] = [
            [0.0] * (state_size + 1) for _ in range(num_actions)
        ]
        self.weights_b: list[list[float]] = [
            [0.0] * (state_size + 1) for _ in range(num_actions)
        ]

    @property
    def weights(self) -> list[list[float]]:
        """Return the combined (averaged) weights of both estimators for serialization."""
        return [
            [(wa + wb) / 2.0 for wa, wb in zip(row_a, row_b)]
            for row_a, row_b in zip(self.weights_a, self.weights_b)
        ]

    @weights.setter
    def weights(self, new_weights: list[list[float]]) -> None:
        """Set weights for both estimators."""
        self.weights_a = [list(row) for row in new_weights]
        self.weights_b = [list(row) for row in new_weights]

    def _with_bias(self, state: list[float]) -> list[float]:
        """Append a constant bias feature (1.0)."""
        if len(state) != self.state_size:
            raise ValueError(
                f"Expected {self.state_size} state features, received {len(state)}"
            )
        for val in state:
            if not math.isfinite(val):
                raise ValueError("State contains non-finite values")
        return [*state, 1.0]

    def q_values(self, state: list[float]) -> list[float]:
        """Estimate the mean value of every action from both estimators."""
        features = self._with_bias(state)
        return [
            sum(((wa + wb) / 2.0) * f for wa, wb, f in zip(row_a, row_b, features))
            for row_a, row_b in zip(self.weights_a, self.weights_b)
        ]

    def choose_action(
        self,
        state: list[float],
        explore: bool = True,
        valid_actions: list[int] | None = None,
    ) -> int:
        """Choose an action using epsilon-greedy selection over valid actions."""
        allowed = valid_actions if valid_actions else list(range(self.num_actions))
        if not allowed:
            return 0

        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(allowed)

        values = self.q_values(state)
        best_value = max(values[a] for a in allowed)
        best_actions = [
            a for a in allowed if math.isclose(values[a], best_value, abs_tol=1e-7)
        ]
        # Prefer action 0 (STAY) on tie
        if 0 in best_actions and len(best_actions) > 1:
            return 0
        return self.rng.choice(best_actions)

    def decay_epsilon(self) -> None:
        """Decay exploration rate toward minimum threshold."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        done: bool = False,
    ) -> float:
        """Update weights using Double Q-learning target with error clipping."""
        if not 0 <= action < self.num_actions:
            raise ValueError(f"Action index {action} is out of range")
        if not math.isfinite(reward):
            raise ValueError(f"Reward {reward} is not finite")

        features = self._with_bias(state)
        next_features = self._with_bias(next_state)

        # Randomly choose which estimator to update (Double Q-learning)
        if self.rng.random() < 0.5:
            # Update A using B for evaluation
            current_q = sum(w * f for w, f in zip(self.weights_a[action], features))
            if done:
                target = reward
            else:
                q_next_a = [
                    sum(w * f for w, f in zip(row, next_features))
                    for row in self.weights_a
                ]
                best_action = q_next_a.index(max(q_next_a))
                q_next_b_val = sum(
                    w * f for w, f in zip(self.weights_b[best_action], next_features)
                )
                target = reward + self.discount_factor * q_next_b_val

            td_error = target - current_q
            clipped_error = max(-self.max_td_error, min(self.max_td_error, td_error))

            for idx, feat in enumerate(features):
                new_w = (
                    self.weights_a[action][idx]
                    + self.learning_rate * clipped_error * feat
                )
                self.weights_a[action][idx] = max(
                    -self.max_weight, min(self.max_weight, new_w)
                )
        else:
            # Update B using A for evaluation
            current_q = sum(w * f for w, f in zip(self.weights_b[action], features))
            if done:
                target = reward
            else:
                q_next_b = [
                    sum(w * f for w, f in zip(row, next_features))
                    for row in self.weights_b
                ]
                best_action = q_next_b.index(max(q_next_b))
                q_next_a_val = sum(
                    w * f for w, f in zip(self.weights_a[best_action], next_features)
                )
                target = reward + self.discount_factor * q_next_a_val

            td_error = target - current_q
            clipped_error = max(-self.max_td_error, min(self.max_td_error, td_error))

            for idx, feat in enumerate(features):
                new_w = (
                    self.weights_b[action][idx]
                    + self.learning_rate * clipped_error * feat
                )
                self.weights_b[action][idx] = max(
                    -self.max_weight, min(self.max_weight, new_w)
                )

        return td_error
