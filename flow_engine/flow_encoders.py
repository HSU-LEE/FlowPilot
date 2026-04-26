from __future__ import annotations



from dataclasses import dataclass, field



from .flow_core import EncodedSample, FlowEncoder, FlowSample





@dataclass

class TimeSeriesEncoder:

    """Default time-series aware encoder plugin."""



    base: FlowEncoder = field(default_factory=FlowEncoder)



    def encode(self, sample: FlowSample) -> EncodedSample:

        return self.base.encode(sample)





@dataclass

class EventLogEncoder:

    """Event-focused encoder with denser categorical signals."""



    base: FlowEncoder = field(default_factory=lambda: FlowEncoder(max_numeric_features=8))



    def encode(self, sample: FlowSample) -> EncodedSample:

        encoded = self.base.encode(sample)

        encoded.flow_features["event_density"] = (

            encoded.flow_features.get("event_count", 0.0) / max(1.0, encoded.flow_features.get("time_span", 1.0))

        )

        encoded.vector.append(encoded.flow_features["event_density"])

        return encoded

