from __future__ import annotations



from dataclasses import dataclass, field

from typing import Any, Iterable, Optional



from .flow_core import FlowSample

from .flow_product import FlowPilot





@dataclass

class StepSignal:

    """Runtime feedback signal from external training loops."""



    loss: float = 0.0

    accuracy: float = 0.0

    confidence: float = 0.0

    grad_norm: float = 0.0



    def to_logs(self) -> dict[str, float]:

        logs: dict[str, float] = {}

        if self.loss > 0.0:

            logs["loss"] = self.loss

        if self.accuracy > 0.0:

            logs["accuracy"] = self.accuracy

        if self.confidence > 0.0:

            logs["confidence"] = self.confidence

        if self.grad_norm > 0.0:

            logs["grad_norm"] = self.grad_norm

        return logs





@dataclass

class FlowBatchPolicy:

    """
    DataLoader-level adapter for on-the-fly batch selection.

    Typical usage:
        policy = FlowBatchPolicy(pilot, samples)
        batch = policy.next_batch(32)
        policy.on_step_end(StepSignal(loss=...))
    """



    pilot: FlowPilot

    samples: list[FlowSample]
    _sample_index: dict[str, FlowSample] = field(init=False, repr=False, default_factory=dict)



    def __post_init__(self) -> None:

        self.pilot.prepare(self.samples)
        self._sample_index = {sample.sample_id: sample for sample in self.samples}



    def next_batch(self, batch_size: int = 32, strategy: Optional[str] = None) -> list[FlowSample]:

        selected = self.pilot.select_next(batch_size=batch_size, strategy=strategy)
        batch: list[FlowSample] = []
        for item in selected:
            source = self._sample_index.get(item.sample.sample_id)
            if source is not None:
                batch.append(source)
        return batch



    def batch_ids(self, batch_size: int = 32, strategy: Optional[str] = None) -> list[str]:

        return [sample.sample_id for sample in self.next_batch(batch_size=batch_size, strategy=strategy)]



    def on_step_end(self, signal: StepSignal | dict[str, Any]) -> None:

        logs = signal.to_logs() if isinstance(signal, StepSignal) else dict(signal)

        self.pilot.trainer.update_state_from_metrics(logs)



    def iter_batches(

        self,

        steps: int,

        batch_size: int = 32,

        strategy: Optional[str] = None,

    ) -> Iterable[list[FlowSample]]:

        for _ in range(max(1, int(steps))):

            yield self.next_batch(batch_size=batch_size, strategy=strategy)

