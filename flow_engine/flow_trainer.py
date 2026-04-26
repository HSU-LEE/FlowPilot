from __future__ import annotations



from dataclasses import dataclass, field

from typing import Any
import math



from .flow_core import EncodedSample, FlowEncoder, FlowSample



from .flow_difficulty import DifficultyEstimator

from .flow_learn import BaselineFlowModel, LearnableModel

from .flow_runtime import ModelState

from .flow_sampler import AdaptiveSampler

from .flow_selector import FlowSelector, SelectionItem





@dataclass

class ScheduleLog:

    step: int

    strategy: str

    selected_sample_ids: list[str]

    avg_difficulty: float

    avg_confidence: float

    reason: str





@dataclass

class TrainingReport:

    steps: int

    schedule_logs: list[ScheduleLog] = field(default_factory=list)

    final_confidence: float = 0.0





@dataclass

class FlowTrainer:

    encoder: Any = field(default_factory=FlowEncoder)

    difficulty: DifficultyEstimator = field(default_factory=DifficultyEstimator)

    selector: FlowSelector = field(init=False)

    sampler: AdaptiveSampler = field(default_factory=AdaptiveSampler)

    model: LearnableModel = field(default_factory=BaselineFlowModel)

    model_state: ModelState = field(default_factory=ModelState)



    def __post_init__(self) -> None:

        self.selector = FlowSelector(estimator=self.difficulty)



    def fit(self, data: list[FlowSample], epochs: int = 1, batch_size: int = 16) -> TrainingReport:

        encoded = [self.encoder.encode(sample) for sample in data]

        logs: list[ScheduleLog] = []

        step = 0



        for _ in range(max(1, epochs)):

            while True:

                decision, selected = self.select_next_batch(encoded, batch_size)

                if not selected:

                    break

                batch = [item.sample for item in selected]

                self.model.train_batch(batch)

                step += 1

                prev_conf = self.model_state.avg_confidence

                self._update_state(batch)

                reward = self.model_state.avg_confidence - prev_conf

                self.model_state.record_strategy_outcome(decision.strategy, reward)

                logs.append(

                    ScheduleLog(

                        step=step,

                        strategy=decision.strategy,

                        selected_sample_ids=[item.sample.sample_id for item in selected],

                        avg_difficulty=self._avg_difficulty(selected),

                        avg_confidence=self.model_state.avg_confidence,

                        reason=decision.reason,

                    )

                )

                if step >= max(1, len(encoded)):

                    break

            if step >= max(1, len(encoded)):

                break



        return TrainingReport(steps=step, schedule_logs=logs, final_confidence=self.model_state.avg_confidence)



    def select_next_batch(

        self, encoded_samples: list[EncodedSample], batch_size: int

    ) -> tuple[Any, list[SelectionItem]]:

        decision = self.sampler.decide(self.model_state)

        selected = self.selector.select(

            encoded_samples=encoded_samples,

            model_state=self.model_state,

            batch_size=batch_size,

            strategy=decision.strategy,

        )

        return decision, selected



    def predict(self, sample: FlowSample) -> Any:

        encoded = self.encoder.encode(sample)

        return self.model.predict(encoded)



    def explain_selection(self, sample: FlowSample) -> dict[str, float]:

        encoded = self.encoder.encode(sample)

        score = self.difficulty.score(encoded, self.model_state)

        return {

            "difficulty_total": score.total,

            "uncertainty": score.uncertainty,

            "volatility": score.volatility,

            "novelty": score.novelty,

            "error_history": score.error_history,
            "loss": score.loss,

        }



    def _update_state(self, batch: list[EncodedSample]) -> None:

        confidences = [self.model.predict_confidence(item) for item in batch]

        self.update_state_from_confidences(confidences)

        for item, conf in zip(batch, confidences):

            self.model_state.sample_uncertainty[item.sample_id] = 1.0 - conf

            self.model_state.sample_error_history[item.sample_id] = max(0.0, 1.0 - conf)
            self.model_state.sample_loss_history[item.sample_id] = min(1.0, max(0.0, -math.log(max(1e-6, conf)) / 2.5))

            correctness = 1.0 - self.model_state.sample_error_history[item.sample_id]

            self.model_state.record_calibration_pair(conf, correctness)

            self.model_state.update_label_confidence(item.label, conf)

            cluster = self.selector.cluster_backend.assign(item)

            prev_cluster = self.model_state.cluster_confidence.get(cluster, self.model_state.avg_confidence)

            self.model_state.cluster_confidence[cluster] = 0.7 * prev_cluster + 0.3 * conf

        self._update_embedding_stats(batch)



    def update_state_from_confidences(self, confidences: list[float]) -> None:

        if not confidences:

            return

        avg_conf = sum(confidences) / len(confidences)

        self.model_state.step += 1

        self.model_state.avg_confidence = 0.7 * self.model_state.avg_confidence + 0.3 * avg_conf

        self.model_state.rolling_error_rate = max(0.0, 1.0 - self.model_state.avg_confidence)

        self.model_state.calibration_error = self.model_state.compute_ece()



    def update_state_from_metrics(self, logs: dict[str, Any]) -> None:

        if "confidence" in logs:

            conf = float(logs["confidence"])

            self.update_state_from_confidences([max(0.0, min(1.0, conf))])

            return



        if "accuracy" in logs:

            acc = float(logs["accuracy"])

            self.update_state_from_confidences([max(0.0, min(1.0, acc))])

            return



        if "loss" in logs:

            loss = max(0.0, float(logs["loss"]))

            conf = 1.0 / (1.0 + loss)

            self.update_state_from_confidences([conf])
            if "grad_norm" in logs:
                grad = max(0.0, float(logs["grad_norm"]))
                penalty = min(0.2, grad / 100.0)
                self.model_state.rolling_error_rate = min(1.0, self.model_state.rolling_error_rate + penalty)



    @staticmethod

    def _avg_difficulty(items: list[SelectionItem]) -> float:

        if not items:

            return 0.0

        return sum(item.difficulty.total for item in items) / len(items)



    def _update_embedding_stats(self, batch: list[EncodedSample]) -> None:

        vectors = [item.vector for item in batch if item.vector]

        if not vectors:

            return

        width = min(len(v) for v in vectors)

        if width <= 0:

            return

        mean_vector = [sum(v[i] for v in vectors) / len(vectors) for i in range(width)]

        if not self.model_state.embedding_centroid:

            self.model_state.embedding_centroid = mean_vector

        else:

            n = min(width, len(self.model_state.embedding_centroid))

            for i in range(n):

                self.model_state.embedding_centroid[i] = (

                    0.8 * self.model_state.embedding_centroid[i] + 0.2 * mean_vector[i]

                )

        avg_norm = sum(sum(abs(v_i) for v_i in vec[:width]) / width for vec in vectors) / len(vectors)

        self.model_state.embedding_scale = max(1e-6, 0.8 * self.model_state.embedding_scale + 0.2 * avg_norm)



