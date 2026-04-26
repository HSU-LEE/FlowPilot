from __future__ import annotations



from dataclasses import dataclass

import math



from .flow_core import EncodedSample

from .flow_runtime import ModelState





def _clamp01(value: float) -> float:

    return max(0.0, min(1.0, value))





@dataclass

class DifficultyScore:

    total: float

    uncertainty: float

    volatility: float

    novelty: float

    error_history: float
    loss: float





@dataclass

class DifficultyEstimator:

    w_uncertainty: float = 0.35

    w_volatility: float = 0.05

    w_novelty: float = 0.05

    w_error_history: float = 0.10
    w_loss: float = 0.45

    temperature: float = 1.5



    def score(self, sample: EncodedSample, model_state: ModelState) -> DifficultyScore:

        raw_uncertainty = model_state.sample_uncertainty.get(sample.sample_id, 1.0 - model_state.avg_confidence)

        uncertainty = self._calibrated_uncertainty(raw_uncertainty, model_state.calibration_error)

        volatility = self._volatility_score(sample)

        novelty = self._novelty_score(sample, model_state)

        error_history = _clamp01(model_state.sample_error_history.get(sample.sample_id, model_state.rolling_error_rate))
        loss_term = _clamp01(model_state.sample_loss_history.get(sample.sample_id, error_history))

        model_state.update_metric("uncertainty", uncertainty)

        model_state.update_metric("volatility", volatility)

        model_state.update_metric("novelty", novelty)

        model_state.update_metric("error_history", error_history)
        model_state.update_metric("loss", loss_term)



        uncertainty = model_state.normalize_metric("uncertainty", uncertainty)

        volatility = model_state.normalize_metric("volatility", volatility)

        novelty = model_state.normalize_metric("novelty", novelty)

        error_history = model_state.normalize_metric("error_history", error_history)
        loss_term = model_state.normalize_metric("loss", loss_term)



        total = (

            uncertainty * self.w_uncertainty

            + volatility * self.w_volatility

            + novelty * self.w_novelty

            + error_history * self.w_error_history
            + loss_term * self.w_loss

        )

        return DifficultyScore(

            total=_clamp01(total),

            uncertainty=uncertainty,

            volatility=volatility,

            novelty=novelty,

            error_history=error_history,
            loss=loss_term,

        )



    def _calibrated_uncertainty(self, uncertainty: float, calibration_error: float) -> float:

        value = _clamp01(uncertainty)

        spread = 1.0 + min(0.75, max(0.0, calibration_error))
        return _clamp01(0.5 + (value - 0.5) * spread)



    @staticmethod

    def _volatility_score(sample: EncodedSample) -> float:

        values = [v for k, v in sample.flow_features.items() if k.endswith(":volatility")]

        noise = [v for k, v in sample.flow_features.items() if k.endswith(":noise")]

        trend = [abs(v) for k, v in sample.flow_features.items() if k.endswith(":trend")]

        if not values:

            return 0.0

        avg_vol = sum(values) / len(values)

        avg_noise = 0.0 if not noise else sum(noise) / len(noise)

        avg_trend = 0.0 if not trend else sum(trend) / len(trend)

        signal = max(0.0, avg_vol + 0.4 * avg_trend - 0.3 * avg_noise)

        return _clamp01(signal)



    @staticmethod

    def _novelty_score(sample: EncodedSample, model_state: ModelState) -> float:

        if not sample.vector:

            return 0.0

        if not model_state.embedding_centroid:

            magnitude = sum(abs(v) for v in sample.vector) / len(sample.vector)

            return _clamp01(magnitude / 10.0)

        n = min(len(sample.vector), len(model_state.embedding_centroid))

        if n == 0:

            return 0.0

        dist_sq = sum((sample.vector[i] - model_state.embedding_centroid[i]) ** 2 for i in range(n)) / n

        scale = max(1e-6, model_state.embedding_scale)

        normalized = math.sqrt(dist_sq) / scale

        return _clamp01(normalized / 3.0)



