
"""Linear Q-learning agent for ambulance redeployment."""

import random


class LinearQAgent:
    """Learn action values using linear function approximation."""

    def __init__(
        self,
        state_size: int,
        num_actions: int,
        learning_rate: float = 0.01,
        discount_factor: float = 0.95,
        epsilon: float = 0.20,
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
        self.rng = random.Random(seed)

        # One weight vector per action; the final feature is a bias.
        self.weights = [
            [0.0] * (state_size + 1)
            for _ in range(num_actions)
        ]

    def _with_bias(self, state: list[float]) -> list[float]:
        """Append a constant feature so each action has a bias."""
        if len(state) != self.state_size:
            raise ValueError(
                f"Expected {self.state_size} state features, "
                f"received {len(state)}"
            )
        return [*state, 1.0]

    def q_values(self, state: list[float]) -> list[float]:
        """Estimate the value of every available action."""
        features = self._with_bias(state)

        return [
            sum(weight * feature for weight, feature in zip(row, features))
            for row in self.weights
        ]

    def choose_action(
        self,
        state: list[float],
        explore: bool = True,
    ) -> int:
        """Choose an action using epsilon-greedy selection."""
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.num_actions)

        values = self.q_values(state)
        best_value = max(values)
        best_actions = [
            action
            for action, value in enumerate(values)
            if value == best_value
        ]
        return self.rng.choice(best_actions)

    def update(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        done: bool = False,
    ) -> float:
        """Update weights using a one-step Q-learning target."""
        if not 0 <= action < self.num_actions:
            raise ValueError("Action index is out of range")

        features = self._with_bias(state)
        current_q = sum(
            weight * feature
            for weight, feature in zip(self.weights[action], features)
        )

        if done:
            target = reward
        else:
            next_values = self.q_values(next_state)
            target = reward + self.discount_factor * max(next_values)

        td_error = target - current_q

        for index, feature in enumerate(features):
            self.weights[action][index] += (
                self.learning_rate * td_error * feature
            )

        return td_error
