from __future__ import annotations



from dataclasses import dataclass, field

from typing import Any, Union





Numeric = Union[int, float]





@dataclass

class FlowEvent:

    timestamp: float

    entity_id: str

    features: dict[str, Any]

    context: dict[str, Any] = field(default_factory=dict)

    outcome: Any = None





@dataclass

class FlowSample:

    sample_id: str

    events: list[FlowEvent]

    label: Any = None

    metadata: dict[str, Any] = field(default_factory=dict)



    def numeric_series(self, key: str) -> list[float]:

        values: list[float] = []

        for event in self.events:

            value = event.features.get(key)

            if isinstance(value, (int, float)):

                values.append(float(value))

        return values





@dataclass

class EncodedSample:

    sample_id: str

    vector: list[float]

    flow_features: dict[str, float]

    label: Any

    segment_vectors: list[list[float]] = field(default_factory=list)





@dataclass

class FlowSchema:

    required_keys: set[str] = field(default_factory=lambda: {"timestamp", "entity_id", "features"})



    def validate_raw_event(self, raw: dict[str, Any]) -> None:

        missing = self.required_keys - set(raw.keys())

        if missing:

            raise ValueError(f"Raw event missing required keys: {sorted(missing)}")

        if not isinstance(raw["features"], dict):

            raise TypeError("features must be a dictionary")



    def normalize_event(self, raw: dict[str, Any]) -> FlowEvent:

        self.validate_raw_event(raw)

        return FlowEvent(

            timestamp=float(raw["timestamp"]),

            entity_id=str(raw["entity_id"]),

            features=dict(raw["features"]),

            context=dict(raw.get("context", {})),

            outcome=raw.get("outcome"),

        )





@dataclass

class FlowEncoder:

    schema: FlowSchema = field(default_factory=FlowSchema)

    max_numeric_features: int = 16



    def encode(self, sample: FlowSample) -> EncodedSample:

        numeric_keys = self._collect_numeric_keys(sample)

        vector: list[float] = []

        flow_features: dict[str, float] = {}



        segment_vectors: list[list[float]] = []

        for key in numeric_keys:

            series = sample.numeric_series(key)

            if not series:

                continue

            vector.extend(self._aggregate_series(series))

            flow_features[f"{key}:volatility"] = self._volatility(series)

            flow_features[f"{key}:trend"] = self._trend(series)

            flow_features[f"{key}:change_rate"] = self._change_rate(series)

            flow_features[f"{key}:noise"] = self._noise_level(series)

            segment_vectors.extend(self._segment_vectors(series))





        flow_features["event_count"] = float(len(sample.events))

        flow_features["time_span"] = self._time_span(sample)

        flow_features["category_switch_rate"] = self._category_switch_rate(sample)

        flow_features["segment_count"] = float(len(segment_vectors))

        vector.extend(

            [

                flow_features["event_count"],

                flow_features["time_span"],

                flow_features["category_switch_rate"],

            ]

        )



        return EncodedSample(

            sample_id=sample.sample_id,

            vector=vector,

            flow_features=flow_features,

            label=sample.label,

            segment_vectors=segment_vectors,

        )



    def _collect_numeric_keys(self, sample: FlowSample) -> list[str]:

        keys: list[str] = []

        seen: set[str] = set()

        for event in sample.events:

            for key, value in event.features.items():

                if key in seen:

                    continue

                if isinstance(value, (int, float)):

                    keys.append(key)

                    seen.add(key)

                if len(keys) >= self.max_numeric_features:

                    return keys

        return keys



    @staticmethod

    def _aggregate_series(series: list[float]) -> list[float]:

        n = len(series)

        avg = sum(series) / n

        min_v = min(series)

        max_v = max(series)

        last = series[-1]

        return [avg, min_v, max_v, last]



    @staticmethod

    def _volatility(series: list[float]) -> float:

        if len(series) < 2:

            return 0.0

        diffs = [abs(series[i] - series[i - 1]) for i in range(1, len(series))]

        return sum(diffs) / len(diffs)



    @staticmethod

    def _trend(series: list[float]) -> float:

        if len(series) < 2:

            return 0.0

        return (series[-1] - series[0]) / (len(series) - 1)



    @staticmethod

    def _change_rate(series: list[float]) -> float:

        if len(series) < 2:

            return 0.0

        change_count = sum(1 for i in range(1, len(series)) if series[i] != series[i - 1])

        return change_count / (len(series) - 1)



    @staticmethod

    def _noise_level(series: list[float]) -> float:

        if len(series) < 3:

            return 0.0

        trend = FlowEncoder._trend(series)

        detrended = [series[i] - (series[0] + trend * i) for i in range(len(series))]

        return sum(abs(v) for v in detrended) / len(detrended)



    @staticmethod

    def _segment_vectors(series: list[float], window: int = 3) -> list[list[float]]:

        if len(series) < window:

            if not series:

                return []

            return [FlowEncoder._aggregate_series(series)]

        vectors: list[list[float]] = []

        for idx in range(0, len(series) - window + 1):

            chunk = series[idx : idx + window]

            vectors.append(FlowEncoder._aggregate_series(chunk))

        return vectors



    @staticmethod

    def _time_span(sample: FlowSample) -> float:

        if len(sample.events) < 2:

            return 0.0

        return sample.events[-1].timestamp - sample.events[0].timestamp



    @staticmethod

    def _category_switch_rate(sample: FlowSample) -> float:

        if len(sample.events) < 2:

            return 0.0

        switches = 0

        comparisons = 0

        prev = sample.events[0].features

        for event in sample.events[1:]:

            for key, value in event.features.items():

                if isinstance(value, str) and key in prev and isinstance(prev[key], str):

                    comparisons += 1

                    if prev[key] != value:

                        switches += 1

            prev = event.features

        return 0.0 if comparisons == 0 else switches / comparisons

