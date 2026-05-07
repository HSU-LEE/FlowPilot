# FlowPilot

FlowPilot is a Pre-ML optimization module for training data scheduling.
It does not replace your model. It optimizes data order and batch selection before and during training.

Core goals:
- reduce training time
- improve data efficiency
- reduce convergence steps

> Before you train, we decide what to learn.

## What this module does

Most training pipelines rely on random sampling.
FlowPilot dynamically selects the next batch based on model state and flow-aware signals.

- difficulty estimation: uncertainty, volatility, novelty, error history
- strategy selection: uncertainty-first, mixed, easy-to-hard, temporal-flow
- policy adaptation: bandit-based explore/exploit
- evidence reporting: baseline-aware KPIs with statistical metrics

## When to use FlowPilot. 

- large datasets with repeated training cycles
- time-ordered data such as logs, sensors, and event streams
- teams where training lead time matters as much as final quality

## Important constraints

- impact depends on dataset, model, and training setup
- this is not a generic AutoML tool for raw image/text inputs
- you still need an existing trainable model

## Installation

```bash
pip install .
```

Optional TensorFlow integration:

```bash
pip install ".[tensorflow]"
```

Editable install (development mode):

```bash
pip install -e .
```

## Quick start (Convenience API)

```python
import flowpilot as fp

data = [
    {
        "sample_id": "s1",
        "events": [
            {"timestamp": 0, "entity_id": "u1", "features": {"value": 1.0}},
            {"timestamp": 1, "entity_id": "u1", "features": {"value": 1.8}},
        ],
        "label": 1,
    }
]

result = fp.run(
    data=data,
    mode="auto",
    target="classification",
    baseline_strategy="random",
    runs=5,
    seed=42,
)

print(result.kpi.training_time_reduction_pct)
print(result.kpi.data_efficiency_gain_pct)
print(result.kpi.convergence_step_reduction_x)
print(result.statistics["time_reduction_ci95"])
print(result.statistics["time_reduction_p_value"])
```

## Recommended usage (Object API)

```python
import flowpilot as fp

pilot = fp.FlowPilot(encoder=fp.TimeSeriesEncoder())
pilot.prepare(data)
batch = pilot.select_next(batch_size=32)
train_logs = pilot.train(steps=10, batch_size=32)
factors = pilot.explain("s1")
```

## DataLoader-style integration

```python
import flowpilot as fp

pilot = fp.FlowPilot()
policy = fp.FlowBatchPolicy(pilot=pilot, samples=data)

batch = policy.next_batch(batch_size=32)
policy.on_step_end(fp.StepSignal(loss=0.42, accuracy=0.81, grad_norm=1.7))
```

## Benchmark and release gate

```python
import flowpilot as fp

pilot = fp.FlowPilot()
harness = fp.BenchmarkHarness(runs=5, seed=42)
results = harness.run(pilot=pilot, data=data, epochs=1, batch_size=32)

for r in results:
    print(r.baseline, r.kpi, r.statistics["time_reduction_ci95"])
    print("gate:", fp.BenchmarkHarness.release_gate_passed(r))
```

Baseline suite:
- `random`
- `uncertainty_only`
- `curriculum_only`

Reported statistics:
- standard deviation (`std`)
- 95% confidence interval (`ci95`)
- paired significance approximation (`p_value`)
- reproducibility manifest (`seed_list`, `config_hash`, `dataset_fingerprint`)

## Encoder plugins

- `fp.TimeSeriesEncoder()`
- `fp.EventLogEncoder()`

You can inject a domain-specific encoder via `FlowPilot(encoder=...)`.

## TensorFlow integration

```python
from flow_engine.flow_tensorflow import CurriculumCallback
from flow_engine.flow_trainer import FlowTrainer

trainer = FlowTrainer()
callback = CurriculumCallback(trainer, monitor="loss", update_every_n_batches=1)

model.fit(x_train, y_train, epochs=10, callbacks=[callback])
print(callback.get_curriculum_state())
```

## Version

```python
import flowpilot as fp
print(fp.__version__)
```
