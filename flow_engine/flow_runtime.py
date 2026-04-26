from __future__ import annotations



from dataclasses import dataclass, field

from typing import Any





@dataclass

class ModelState:

    step: int = 0

    avg_confidence: float = 0.5

    target_confidence: float = 0.8

    rolling_error_rate: float = 0.5

    sample_error_history: dict[str, float] = field(default_factory=dict)
    sample_loss_history: dict[str, float] = field(default_factory=dict)

    sample_uncertainty: dict[str, float] = field(default_factory=dict)

    label_confidence: dict[str, float] = field(default_factory=dict)

    cluster_confidence: dict[int, float] = field(default_factory=dict)

    calibration_error: float = 0.0

    embedding_centroid: list[float] = field(default_factory=list)

    embedding_scale: float = 1.0

    strategy_counts: dict[str, int] = field(default_factory=dict)

    strategy_rewards: dict[str, float] = field(default_factory=dict)

    strategy_reward_history: dict[str, list[float]] = field(default_factory=dict)

    recent_losses: list[float] = field(default_factory=list)

    metric_history: dict[str, list[float]] = field(default_factory=dict)

    normalizer_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    calibration_bins: list[tuple[float, float]] = field(default_factory=list)



    def phase(self) -> str:

        if self.step < 100:

            return "warmup"

        if self.avg_confidence < self.target_confidence:

            return "improve"

        return "stabilize"



    def record_strategy_outcome(self, strategy: str, reward: float, window: int = 16) -> None:

        self.strategy_counts[strategy] = self.strategy_counts.get(strategy, 0) + 1

        history = self.strategy_reward_history.setdefault(strategy, [])

        history.append(reward)

        if len(history) > window:

            del history[0 : len(history) - window]

        avg = sum(history) / len(history)

        var = 0.0

        if len(history) > 1:

            var = sum((item - avg) ** 2 for item in history) / (len(history) - 1)

        penalty = 0.1 * (var**0.5)

        self.strategy_rewards[strategy] = avg - penalty



    def update_label_confidence(self, label: Any, confidence: float) -> None:

        key = str(label)

        prev = self.label_confidence.get(key, self.avg_confidence)

        self.label_confidence[key] = 0.7 * prev + 0.3 * confidence



    def update_metric(self, name: str, value: float, history_limit: int = 128) -> None:

        history = self.metric_history.setdefault(name, [])

        history.append(value)

        if len(history) > history_limit:

            del history[0 : len(history) - history_limit]

        sorted_vals = sorted(history)

        low = sorted_vals[max(0, int(0.1 * (len(sorted_vals) - 1)))]

        high = sorted_vals[max(0, int(0.9 * (len(sorted_vals) - 1)))]

        if high <= low:

            high = low + 1e-6

        self.normalizer_bounds[name] = (low, high)



    def normalize_metric(self, name: str, value: float) -> float:

        bounds = self.normalizer_bounds.get(name)

        if bounds is None:

            return max(0.0, min(1.0, value))

        low, high = bounds

        scaled = (value - low) / max(1e-6, high - low)

        return max(0.0, min(1.0, scaled))



    def record_calibration_pair(self, confidence: float, correctness: float, max_pairs: int = 256) -> None:

        self.calibration_bins.append((max(0.0, min(1.0, confidence)), max(0.0, min(1.0, correctness))))

        if len(self.calibration_bins) > max_pairs:

            del self.calibration_bins[0 : len(self.calibration_bins) - max_pairs]



    def compute_ece(self, bucket_count: int = 10) -> float:

        if not self.calibration_bins:

            return 0.0

        buckets: list[list[tuple[float, float]]] = [[] for _ in range(bucket_count)]

        for conf, corr in self.calibration_bins:

            idx = min(bucket_count - 1, int(conf * bucket_count))

            buckets[idx].append((conf, corr))

        total = len(self.calibration_bins)

        ece = 0.0

        for bucket in buckets:

            if not bucket:

                continue

            avg_conf = sum(item[0] for item in bucket) / len(bucket)

            avg_corr = sum(item[1] for item in bucket) / len(bucket)

            ece += abs(avg_conf - avg_corr) * (len(bucket) / total)

        return ece

