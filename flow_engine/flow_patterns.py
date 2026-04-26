from __future__ import annotations



from dataclasses import dataclass, field

from typing import Iterable, Sequence





def _safe_div(numerator: float, denominator: float) -> float:

    if abs(denominator) < 1e-12:

        return 0.0

    return numerator / denominator





@dataclass

class WeightedTrendPredictor:

    """
    Domain-neutral weighted trend predictor.

    This is adapted from the weighted short-horizon prediction pattern in `test.py`
    (`_weighted_ball_velocity` + `predict_position`) but works for any numeric sequence.
    """



    weights: Sequence[float]

    horizon_scale: float = 1.0



    def predict_next(self, values: Sequence[float]) -> float:

        if len(values) < 2:

            return values[-1] if values else 0.0



        deltas = [values[i] - values[i - 1] for i in range(1, len(values))]

        recent = deltas[-len(self.weights) :]

        use_weights = self.weights[-len(recent) :]

        weighted_delta = sum(d * w for d, w in zip(recent, use_weights))

        return values[-1] + weighted_delta * self.horizon_scale



    def trend_strength(self, values: Sequence[float]) -> float:

        if len(values) < 2:

            return 0.0

        delta_sum = sum(abs(values[i] - values[i - 1]) for i in range(1, len(values)))

        return _safe_div(delta_sum, len(values) - 1)





@dataclass

class StateTransitionPolicy:

    """
    Generic state transition controller.

    Mirrors the role/state flags from `test.py` with a simple transition table.
    """



    initial_state: str

    transitions: dict[str, dict[str, str]] = field(default_factory=dict)

    terminal_states: set[str] = field(default_factory=set)



    def next_state(self, current_state: str, event: str) -> str:

        if current_state in self.terminal_states:

            return current_state

        state_rules = self.transitions.get(current_state, {})

        return state_rules.get(event, current_state)



    def run(self, events: Iterable[str]) -> str:

        state = self.initial_state

        for event in events:

            state = self.next_state(state, event)

        return state





@dataclass

class SafetyConstraint:

    name: str

    threshold: float

    direction: str = "min"



    def violated(self, value: float) -> bool:

        if self.direction == "min":

            return value < self.threshold

        if self.direction == "max":

            return value > self.threshold

        raise ValueError(f"Unsupported direction: {self.direction}")





@dataclass

class SafetyConstraintSet:

    """
    Generic safety gate inspired by collision/cluster constraints in `test.py`.
    """



    constraints: list[SafetyConstraint] = field(default_factory=list)



    def add(self, constraint: SafetyConstraint) -> None:

        self.constraints.append(constraint)



    def evaluate(self, metrics: dict[str, float]) -> dict[str, bool]:

        result: dict[str, bool] = {}

        for c in self.constraints:

            value = metrics.get(c.name)

            result[c.name] = False if value is None else c.violated(value)

        return result



    def is_safe(self, metrics: dict[str, float]) -> bool:

        violations = self.evaluate(metrics)

        return not any(violations.values())

