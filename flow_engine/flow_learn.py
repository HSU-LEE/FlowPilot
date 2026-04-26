from __future__ import annotations



from dataclasses import dataclass, field

from typing import Any



from .flow_core import EncodedSample





class LearnableModel:

    def train_batch(self, batch: list[EncodedSample]) -> None:

        raise NotImplementedError



    def predict_confidence(self, sample: EncodedSample) -> float:

        raise NotImplementedError



    def predict(self, sample: EncodedSample) -> Any:

        raise NotImplementedError





@dataclass

class BaselineFlowModel:

    """
    Lightweight baseline model with no third-party dependencies.

    Uses nearest centroid style confidence: if encoded vectors are close to the
    running centroid, confidence increases.
    """



    centroid: list[float] = field(default_factory=list)

    seen: int = 0



    def train_batch(self, batch: list[EncodedSample]) -> None:

        for sample in batch:

            self._update_centroid(sample.vector)



    def predict_confidence(self, sample: EncodedSample) -> float:

        if not self.centroid or not sample.vector:

            return 0.5

        n = min(len(self.centroid), len(sample.vector))

        if n == 0:

            return 0.5

        dist = sum(abs(self.centroid[i] - sample.vector[i]) for i in range(n)) / n



        return max(0.0, min(1.0, 1.0 / (1.0 + dist)))



    def predict(self, sample: EncodedSample) -> Any:

        return sample.label



    def _update_centroid(self, vector: list[float]) -> None:

        if not vector:

            return

        if not self.centroid:

            self.centroid = list(vector)

            self.seen = 1

            return

        n = min(len(self.centroid), len(vector))

        self.seen += 1

        alpha = 1.0 / self.seen

        for i in range(n):

            self.centroid[i] = self.centroid[i] * (1 - alpha) + vector[i] * alpha

