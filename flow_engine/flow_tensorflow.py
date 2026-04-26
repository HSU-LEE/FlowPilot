from __future__ import annotations



from dataclasses import dataclass, field

from typing import Any, Optional



from .flow_core import EncodedSample, FlowSample

from .flow_trainer import FlowTrainer



try:

    import tensorflow as tf



    CallbackBase = tf.keras.callbacks.Callback

except Exception:

    tf = None



    class CallbackBase(object):

        pass





@dataclass

class TensorFlowFlowBridge:

    """
    Helper that lets TensorFlow training loops request curriculum decisions.

    This can be used inside custom training loops or Keras generators.
    """



    trainer: FlowTrainer

    encoded_cache: dict[str, EncodedSample] = field(default_factory=dict)



    def encode_dataset(self, samples: list[FlowSample]) -> list[EncodedSample]:

        encoded: list[EncodedSample] = []

        for sample in samples:

            item = self.trainer.encoder.encode(sample)

            self.encoded_cache[sample.sample_id] = item

            encoded.append(item)

        return encoded



    def next_batch_ids(

        self, samples: list[FlowSample], batch_size: int, strategy: Optional[str] = None

    ) -> list[str]:

        encoded = self.encode_dataset(samples)

        if strategy:

            selected = self.trainer.selector.select(

                encoded_samples=encoded,

                model_state=self.trainer.model_state,

                batch_size=batch_size,

                strategy=strategy,

            )

        else:

            _decision, selected = self.trainer.select_next_batch(encoded, batch_size)

        return [item.sample.sample_id for item in selected]



    def update_from_keras_logs(self, logs: Optional[dict[str, Any]]) -> None:

        if not logs:

            return

        self.trainer.update_state_from_metrics(logs)





class CurriculumCallback(CallbackBase):

    """
    Keras callback that updates FlowTrainer state during epochs/batches.

    - on_train_batch_end: adapt curriculum confidence from live metrics
    - on_epoch_begin: decide preferred strategy for this epoch
    """



    def __init__(

        self,

        trainer: FlowTrainer,

        monitor: str = "loss",

        update_every_n_batches: int = 1,

    ) -> None:

        super(CurriculumCallback, self).__init__()

        self.trainer = trainer

        self.monitor = monitor

        self.update_every_n_batches = max(1, int(update_every_n_batches))

        self.batch_count = 0

        self.current_strategy = "mixed"

        self.last_reason = "initial"



    def on_epoch_begin(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:

        del epoch

        decision = self.trainer.sampler.decide(self.trainer.model_state)

        self.current_strategy = decision.strategy

        self.last_reason = decision.reason



    def on_train_batch_end(self, batch: int, logs: Optional[dict[str, Any]] = None) -> None:

        del batch

        self.batch_count += 1

        if self.batch_count % self.update_every_n_batches != 0:

            return

        if logs and self.monitor in logs:

            self.trainer.update_state_from_metrics({self.monitor: logs[self.monitor]})

        else:

            self.trainer.update_state_from_metrics(logs or {})



    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:

        del epoch

        self.trainer.update_state_from_metrics(logs or {})



    def get_curriculum_state(self) -> dict[str, Any]:

        return {

            "strategy": self.current_strategy,

            "reason": self.last_reason,

            "avg_confidence": self.trainer.model_state.avg_confidence,

            "step": self.trainer.model_state.step,

        }

