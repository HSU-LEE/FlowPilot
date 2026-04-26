from __future__ import annotations



from dataclasses import dataclass, field

from time import perf_counter, time

from typing import Any, Optional

import random

import math

import json

import hashlib



from .flow_core import EncodedSample, FlowEncoder, FlowEvent, FlowSample

from .flow_selector import SelectionItem

from .flow_trainer import FlowTrainer





def _safe_div(numerator: float, denominator: float) -> float:

    if denominator == 0:

        return 0.0

    return numerator / denominator





@dataclass

class FitKPI:

    training_time_reduction_pct: float

    data_efficiency_gain_pct: float

    convergence_step_reduction_x: float

    optimized_training_time_sec: float

    baseline_training_time_sec: float

    optimized_data_used: int

    baseline_data_used: int

    optimized_convergence_step: int

    baseline_convergence_step: int

    baseline: str





@dataclass

class VisualizationSeries:

    difficulty_over_time: list[float] = field(default_factory=list)

    strategy_over_time: list[str] = field(default_factory=list)

    confidence_over_time: list[float] = field(default_factory=list)

    loss_over_time: list[float] = field(default_factory=list)

    selected_data_flow: list[list[str]] = field(default_factory=list)





@dataclass

class FitResult:

    message: str

    kpi: FitKPI

    visualization: VisualizationSeries

    summary: dict[str, Any] = field(default_factory=dict)

    statistics: dict[str, Any] = field(default_factory=dict)



    @property

    def logs(self) -> list[dict[str, Any]]:

        logs: list[dict[str, Any]] = []

        for idx, selected in enumerate(self.visualization.selected_data_flow):

            logs.append(

                {

                    "step": idx + 1,

                    "strategy": self.visualization.strategy_over_time[idx]

                    if idx < len(self.visualization.strategy_over_time)

                    else "unknown",

                    "selected_sample_ids": selected,

                    "difficulty": self.visualization.difficulty_over_time[idx]

                    if idx < len(self.visualization.difficulty_over_time)

                    else 0.0,

                    "confidence": self.visualization.confidence_over_time[idx]

                    if idx < len(self.visualization.confidence_over_time)

                    else 0.0,

                    "loss_proxy": self.visualization.loss_over_time[idx]

                    if idx < len(self.visualization.loss_over_time)

                    else 0.0,

                }

            )

        return logs



    @property

    def charts(self) -> dict[str, Any]:

        return {

            "difficulty_over_time": self.visualization.difficulty_over_time,

            "strategy_over_time": self.visualization.strategy_over_time,

            "selected_data_flow": self.visualization.selected_data_flow,

            "loss_vs_flow": {

                "loss_over_time": self.visualization.loss_over_time,

                "confidence_over_time": self.visualization.confidence_over_time,

            },

        }



    def to_dict(self) -> dict[str, Any]:

        return {

            "kpi": {

                "training_time_reduction_pct": self.kpi.training_time_reduction_pct,

                "data_efficiency_gain_pct": self.kpi.data_efficiency_gain_pct,

                "convergence_step_reduction_x": self.kpi.convergence_step_reduction_x,

                "baseline": self.kpi.baseline,

                "optimized_training_time_sec": self.kpi.optimized_training_time_sec,

                "baseline_training_time_sec": self.kpi.baseline_training_time_sec,

                "optimized_data_used": self.kpi.optimized_data_used,

                "baseline_data_used": self.kpi.baseline_data_used,

                "optimized_convergence_step": self.kpi.optimized_convergence_step,

                "baseline_convergence_step": self.kpi.baseline_convergence_step,

            },

            "logs": self.logs,

            "charts": self.charts,

            "summary": self.summary,

            "statistics": self.statistics,

            "message": self.message,

        }





@dataclass

class _RunStats:

    elapsed_sec: float

    convergence_step: int

    data_used: int

    difficulty_over_time: list[float]

    strategy_over_time: list[str]

    confidence_over_time: list[float]

    loss_over_time: list[float]

    selected_data_flow: list[list[str]]





@dataclass

class FlowPilot:

    """
    Product facade focused on business outcomes.

    Public contract is intentionally simple:
        flow = FlowPilot()
        result = flow.fit(data)
    """



    target_confidence: float = 0.8

    random_seed: int = 13

    encoder: Optional[Any] = None

    trainer: FlowTrainer = field(init=False)

    prepared_samples: list[FlowSample] = field(default_factory=list)

    prepared_encoded: list[EncodedSample] = field(default_factory=list)



    def __post_init__(self) -> None:

        self.trainer = self._build_trainer(self.random_seed)



    def prepare(self, data: list[Any]) -> list[FlowSample]:

        """Normalize and cache data for step-by-step control."""

        samples = self._normalize_data(data)

        if not samples:

            raise ValueError("data must include at least one sample")

        self.prepared_samples = samples

        self.prepared_encoded = [self.trainer.encoder.encode(sample) for sample in samples]

        return self.prepared_samples



    def train(self, model: Optional[Any] = None, steps: int = 1, batch_size: int = 16) -> list[dict[str, Any]]:

        """
        Train in controlled steps.

        - model: optional external model implementing train_batch/predict_confidence/predict
        - steps: number of scheduling iterations
        """

        if model is not None:

            self.trainer.model = model

        if not self.prepared_encoded:

            raise ValueError("Call prepare(data) before train().")



        logs: list[dict[str, Any]] = []

        for idx in range(max(1, int(steps))):

            selected = self.select_next(batch_size=batch_size)

            if not selected:

                break

            batch = [item.sample for item in selected]

            self.trainer.model.train_batch(batch)

            self.trainer._update_state(batch)

            avg_difficulty = sum(item.difficulty.total for item in selected) / len(selected)

            logs.append(

                {

                    "step": idx + 1,

                    "selected_sample_ids": [item.sample.sample_id for item in selected],

                    "avg_difficulty": avg_difficulty,

                    "avg_confidence": self.trainer.model_state.avg_confidence,

                }

            )

        return logs



    def select_next(self, batch_size: int = 16, strategy: Optional[str] = None) -> list[SelectionItem]:

        """Select next batch from prepared data."""

        if not self.prepared_encoded:

            raise ValueError("Call prepare(data) before select_next().")

        use_strategy = strategy

        if not use_strategy:

            decision = self.trainer.sampler.decide(self.trainer.model_state)

            use_strategy = decision.strategy

        return self.trainer.selector.select(

            encoded_samples=self.prepared_encoded,

            model_state=self.trainer.model_state,

            batch_size=max(1, int(batch_size)),

            strategy=use_strategy,

        )



    def explain(self, sample: Any) -> dict[str, float]:

        """Explain difficulty factors for a sample or sample_id."""

        target_sample: Optional[FlowSample] = None

        if isinstance(sample, FlowSample):

            target_sample = sample

        elif isinstance(sample, str):

            for candidate in self.prepared_samples:

                if candidate.sample_id == sample:

                    target_sample = candidate

                    break

        elif isinstance(sample, dict):

            target_sample = self._flow_sample_from_dict(sample, 0)

        if target_sample is None:

            raise ValueError("sample must be FlowSample, sample_id, or raw sample dict")

        return self.trainer.explain_selection(target_sample)



    def on_step_end(self, logs: dict[str, Any]) -> None:

        """Hook for external training loops to feed live metrics."""

        self.trainer.update_state_from_metrics(logs)



    def create_batch_policy(self, data: list[Any]) -> Any:

        """Create DataLoader-level batch policy adapter."""

        samples = self._normalize_data(data)

        from .flow_dataloader import FlowBatchPolicy



        return FlowBatchPolicy(self, samples)



    def fit(

        self,

        data: list[Any],

        epochs: int = 1,

        batch_size: int = 16,

        baseline_strategy: str = "random",

        mode: str = "auto",

        target: str = "classification",

        runs: int = 3,

        seed: Optional[int] = None,

    ) -> FitResult:

        mode = str(mode).lower()

        if mode not in {"auto", "manual"}:

            raise ValueError("mode must be one of: auto, manual")

        target = str(target).lower()

        if target not in {"classification", "regression", "ranking", "anomaly_detection"}:

            raise ValueError("target must be one of: classification, regression, ranking, anomaly_detection")

        baseline_strategy = str(baseline_strategy).lower()

        if baseline_strategy not in {"random", "mixed", "uncertainty_only", "curriculum_only"}:

            raise ValueError("baseline_strategy must be one of: random, mixed, uncertainty_only, curriculum_only")

        runs = max(1, int(runs))

        run_seed = self.random_seed if seed is None else int(seed)



        samples = self._normalize_data(data)

        if not samples:

            raise ValueError("data must include at least one sample")



        optimized_runs: list[_RunStats] = []

        baseline_runs: list[_RunStats] = []

        for idx in range(runs):

            seed_i = run_seed + idx

            optimized_runs.append(

                self._run(samples=samples, epochs=epochs, batch_size=batch_size, optimized=True, seed=seed_i)

            )

            baseline_runs.append(

                self._run(

                    samples=samples,

                    epochs=epochs,

                    batch_size=batch_size,

                    optimized=False,

                    baseline_strategy=baseline_strategy,

                    seed=seed_i,

                )

            )

        optimized = self._aggregate_runs(optimized_runs)

        baseline = self._aggregate_runs(baseline_runs)

        time_reductions = [

            max(0.0, (b.elapsed_sec - o.elapsed_sec) * 100.0 * _safe_div(1.0, b.elapsed_sec))

            for o, b in zip(optimized_runs, baseline_runs)

        ]

        data_gains = [

            max(0.0, (b.data_used - o.data_used) * 100.0 * _safe_div(1.0, float(b.data_used)))

            for o, b in zip(optimized_runs, baseline_runs)

        ]



        time_reduction = max(

            0.0, (baseline.elapsed_sec - optimized.elapsed_sec) * 100.0 * _safe_div(1.0, baseline.elapsed_sec)

        )

        data_efficiency = max(

            0.0, (baseline.data_used - optimized.data_used) * 100.0 * _safe_div(1.0, float(baseline.data_used))

        )

        step_reduction_x = _safe_div(float(baseline.convergence_step), float(max(1, optimized.convergence_step)))



        kpi = FitKPI(

            training_time_reduction_pct=round(time_reduction, 2),

            data_efficiency_gain_pct=round(data_efficiency, 2),

            convergence_step_reduction_x=round(step_reduction_x, 2),

            optimized_training_time_sec=round(optimized.elapsed_sec, 6),

            baseline_training_time_sec=round(baseline.elapsed_sec, 6),

            optimized_data_used=optimized.data_used,

            baseline_data_used=baseline.data_used,

            optimized_convergence_step=optimized.convergence_step,

            baseline_convergence_step=baseline.convergence_step,

            baseline=baseline_strategy,

        )



        return FitResult(

            message=(

                "FlowPilot optimizes data order before training. "

                "Effects vary by dataset, model, and training setup."

            ),

            kpi=kpi,

            visualization=VisualizationSeries(

                difficulty_over_time=optimized.difficulty_over_time,

                strategy_over_time=optimized.strategy_over_time,

                confidence_over_time=optimized.confidence_over_time,

                loss_over_time=optimized.loss_over_time,

                selected_data_flow=optimized.selected_data_flow,

            ),

            summary={

                "tagline": "Before you train, we decide what to learn.",

                "positioning": "Pre-ML Engine",

                "batch_size": max(1, int(batch_size)),

                "epochs": max(1, int(epochs)),

                "mode": mode,

                "target": target,

                "baseline": baseline_strategy,

                "scope": (

                    "Designed for time-ordered data such as logs, sensor streams, "

                    "and event sequences."

                ),

                "usage": (

                    "Most effective in repeated training loops with enough data "

                    "and time-sensitive iteration cycles."

                ),

                "note": "FlowPilot improves existing model training workflows; it is not a standalone model.",

            },

            statistics={

                "runs": runs,

                "seed": run_seed,

                "manifest": {

                    "generated_at_unix": int(time()),

                    "seed_list": [run_seed + idx for idx in range(runs)],

                    "config_hash": self._config_hash(

                        epochs=epochs,

                        batch_size=batch_size,

                        baseline_strategy=baseline_strategy,

                        mode=mode,

                        target=target,

                    ),

                    "dataset_fingerprint": self._dataset_fingerprint(samples),

                },

                "time_reduction_std": self._std(time_reductions),

                "data_gain_std": self._std(data_gains),

                "time_reduction_ci95": self._ci95(time_reductions),

                "data_gain_ci95": self._ci95(data_gains),

                "time_reduction_p_value": self._paired_p_value(

                    [o.elapsed_sec for o in optimized_runs],

                    [b.elapsed_sec for b in baseline_runs],

                ),

            },

        )



    def _run(

        self,

        samples: list[FlowSample],

        epochs: int,

        batch_size: int,

        optimized: bool,

        baseline_strategy: str = "random",

        seed: Optional[int] = None,

    ) -> _RunStats:

        trainer = self._build_trainer(seed if seed is not None else self.random_seed)

        trainer.model_state.target_confidence = self.target_confidence

        encoded = [trainer.encoder.encode(sample) for sample in samples]

        step_budget = max(1, len(encoded)) * max(1, int(epochs))

        step = 0

        convergence_step: Optional[int] = None

        selected_unique_ids: set[str] = set()

        rng = random.Random(seed if seed is not None else self.random_seed)



        difficulty_over_time: list[float] = []

        strategy_over_time: list[str] = []

        confidence_over_time: list[float] = []

        loss_over_time: list[float] = []

        selected_data_flow: list[list[str]] = []



        start = perf_counter()

        while step < step_budget:

            strategy, selected = self._select_batch(

                trainer=trainer,

                encoded=encoded,

                batch_size=batch_size,

                optimized=optimized,

                baseline_strategy=baseline_strategy,

                rng=rng,

            )

            if not selected:

                break



            batch = [item.sample for item in selected]

            trainer.model.train_batch(batch)

            trainer._update_state(batch)

            step += 1



            avg_diff = 0.0 if not selected else sum(item.difficulty.total for item in selected) / len(selected)

            selected_ids = [item.sample.sample_id for item in selected]

            selected_unique_ids.update(selected_ids)



            difficulty_over_time.append(avg_diff)

            strategy_over_time.append(strategy)

            confidence_over_time.append(trainer.model_state.avg_confidence)

            loss_over_time.append(max(0.0, 1.0 - trainer.model_state.avg_confidence))

            selected_data_flow.append(selected_ids)



            if convergence_step is None and trainer.model_state.avg_confidence >= trainer.model_state.target_confidence:

                convergence_step = step



        elapsed_sec = perf_counter() - start

        return _RunStats(

            elapsed_sec=elapsed_sec,

            convergence_step=convergence_step or step_budget,

            data_used=len(selected_unique_ids),

            difficulty_over_time=difficulty_over_time,

            strategy_over_time=strategy_over_time,

            confidence_over_time=confidence_over_time,

            loss_over_time=loss_over_time,

            selected_data_flow=selected_data_flow,

        )



    def _build_trainer(self, seed: int) -> FlowTrainer:
        encoder = self.encoder if self.encoder is not None else FlowEncoder()
        trainer = FlowTrainer(encoder=encoder)
        trainer.model_state.step = 0
        return trainer



    @staticmethod

    def _aggregate_runs(runs: list[_RunStats]) -> _RunStats:

        if len(runs) == 1:

            return runs[0]

        return _RunStats(

            elapsed_sec=sum(r.elapsed_sec for r in runs) / len(runs),

            convergence_step=int(sum(r.convergence_step for r in runs) / len(runs)),

            data_used=int(sum(r.data_used for r in runs) / len(runs)),

            difficulty_over_time=FlowPilot._mean_series([r.difficulty_over_time for r in runs]),
            strategy_over_time=FlowPilot._majority_series([r.strategy_over_time for r in runs]),
            confidence_over_time=FlowPilot._mean_series([r.confidence_over_time for r in runs]),
            loss_over_time=FlowPilot._mean_series([r.loss_over_time for r in runs]),
            selected_data_flow=runs[0].selected_data_flow,

        )



    @staticmethod

    def _std(values: list[float]) -> float:

        if len(values) < 2:

            return 0.0

        avg = sum(values) / len(values)

        var = sum((v - avg) ** 2 for v in values) / (len(values) - 1)

        return math.sqrt(var)

    @staticmethod
    def _mean_series(series_list: list[list[float]]) -> list[float]:
        if not series_list:
            return []
        max_len = max((len(series) for series in series_list), default=0)
        output: list[float] = []
        for idx in range(max_len):
            values = [series[idx] for series in series_list if idx < len(series)]
            output.append(0.0 if not values else sum(values) / len(values))
        return output

    @staticmethod
    def _majority_series(series_list: list[list[str]]) -> list[str]:
        if not series_list:
            return []
        max_len = max((len(series) for series in series_list), default=0)
        output: list[str] = []
        for idx in range(max_len):
            votes: dict[str, int] = {}
            for series in series_list:
                if idx < len(series):
                    key = series[idx]
                    votes[key] = votes.get(key, 0) + 1
            if not votes:
                output.append("unknown")
            else:
                output.append(max(votes.items(), key=lambda item: item[1])[0])
        return output



    @classmethod

    def _ci95(cls, values: list[float]) -> list[float]:

        if not values:

            return [0.0, 0.0]

        if len(values) < 2:
            mean = sum(values) / len(values)
            return [mean, mean]

        rng = random.Random(1234)
        means: list[float] = []
        n = len(values)
        for _ in range(1000):
            sample = [values[rng.randrange(0, n)] for _ in range(n)]
            means.append(sum(sample) / n)
        means.sort()
        low = means[int(0.025 * (len(means) - 1))]
        high = means[int(0.975 * (len(means) - 1))]
        return [low, high]



    @classmethod

    def _paired_p_value(cls, optimized: list[float], baseline: list[float]) -> float:

        if len(optimized) != len(baseline) or len(optimized) < 2:

            return 1.0

        diffs = [baseline[idx] - optimized[idx] for idx in range(len(optimized))]
        observed = abs(sum(diffs) / len(diffs))
        rng = random.Random(2026)
        exceed = 0
        trials = 2000
        for _ in range(trials):
            signed = [diff if rng.random() > 0.5 else -diff for diff in diffs]
            stat = abs(sum(signed) / len(signed))
            if stat >= observed:
                exceed += 1
        return (exceed + 1) / float(trials + 1)



    @staticmethod

    def _dataset_fingerprint(samples: list[FlowSample]) -> str:

        payload = json.dumps(

            [

                {"id": sample.sample_id, "events": len(sample.events), "label": str(sample.label)}

                for sample in samples

            ],

            sort_keys=True,

        )

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]



    @staticmethod

    def _config_hash(

        epochs: int,

        batch_size: int,

        baseline_strategy: str,

        mode: str,

        target: str,

    ) -> str:

        payload = json.dumps(

            {

                "epochs": epochs,

                "batch_size": batch_size,

                "baseline_strategy": baseline_strategy,

                "mode": mode,

                "target": target,

            },

            sort_keys=True,

        )

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]



    @staticmethod

    def _select_batch(

        trainer: FlowTrainer,

        encoded: list[EncodedSample],

        batch_size: int,

        optimized: bool,

        baseline_strategy: str,

        rng: random.Random,

    ) -> tuple[str, list[SelectionItem]]:

        if optimized:

            decision, selected = trainer.select_next_batch(encoded, max(1, int(batch_size)))

            return decision.strategy, selected



        if baseline_strategy == "mixed":

            selected = trainer.selector.select(

                encoded_samples=encoded,

                model_state=trainer.model_state,

                batch_size=max(1, int(batch_size)),

                strategy="mixed",

            )

            return "mixed", selected

        if baseline_strategy == "uncertainty_only":

            selected = trainer.selector.select(

                encoded_samples=encoded,

                model_state=trainer.model_state,

                batch_size=max(1, int(batch_size)),

                strategy="uncertainty_first",

            )

            return "uncertainty_only", selected

        if baseline_strategy == "curriculum_only":

            selected = trainer.selector.select(

                encoded_samples=encoded,

                model_state=trainer.model_state,

                batch_size=max(1, int(batch_size)),

                strategy="easy_to_hard",

            )

            return "curriculum_only", selected



        shuffled = list(encoded)

        rng.shuffle(shuffled)

        selected_raw = shuffled[: max(1, int(batch_size))]

        selected = [SelectionItem(sample=sample, difficulty=trainer.difficulty.score(sample, trainer.model_state)) for sample in selected_raw]

        return "random", selected



    def _normalize_data(self, data: list[Any]) -> list[FlowSample]:

        normalized: list[FlowSample] = []

        for idx, item in enumerate(data):

            if isinstance(item, FlowSample):

                normalized.append(item)

                continue

            if isinstance(item, dict):

                normalized.append(self._flow_sample_from_dict(item, idx))

                continue

            raise TypeError("data items must be FlowSample or dict")

        return normalized



    @staticmethod

    def _flow_sample_from_dict(item: dict[str, Any], idx: int) -> FlowSample:

        sample_id = str(item.get("sample_id", f"sample-{idx}"))

        events_raw = item.get("events", [])

        if not isinstance(events_raw, list):

            raise TypeError("events must be a list")

        events: list[FlowEvent] = []

        for event_idx, event in enumerate(events_raw):

            if not isinstance(event, dict):

                raise TypeError("each event must be a dictionary")

            features = event.get("features", {})

            if not isinstance(features, dict):

                raise TypeError("event.features must be a dictionary")

            events.append(

                FlowEvent(

                    timestamp=float(event.get("timestamp", event_idx)),

                    entity_id=str(event.get("entity_id", sample_id)),

                    features=features,

                    context=dict(event.get("context", {})),

                    outcome=event.get("outcome"),

                )

            )

        return FlowSample(sample_id=sample_id, events=events, label=item.get("label"), metadata=dict(item.get("metadata", {})))

