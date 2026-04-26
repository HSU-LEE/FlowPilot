from __future__ import annotations

import flowpilot as fp

from flow_engine.flow_core import FlowEvent, FlowSample


def test_next_batch_returns_flow_samples() -> None:
    data = [
        FlowSample(
            sample_id="s1",
            events=[
                FlowEvent(timestamp=0, entity_id="u1", features={"value": 1.0}),
                FlowEvent(timestamp=1, entity_id="u1", features={"value": 1.2}),
            ],
            label=1,
        ),
        FlowSample(
            sample_id="s2",
            events=[
                FlowEvent(timestamp=0, entity_id="u2", features={"value": 0.2}),
                FlowEvent(timestamp=1, entity_id="u2", features={"value": 0.3}),
            ],
            label=0,
        ),
    ]

    pilot = fp.FlowPilot()
    policy = fp.FlowBatchPolicy(pilot=pilot, samples=data)
    batch = policy.next_batch(batch_size=2)

    assert batch
    assert all(isinstance(item, FlowSample) for item in batch)
