import flowpilot as fp
from typing import Any, Dict, List
from flow_engine.flow_tensorflow import CurriculumCallback
from flow_engine.flow_trainer import FlowTrainer
import math
import random
import csv
import os


def make_flow_data(size: int = 2000, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    data = []
    for idx in range(size):
        hard_zone = idx % 4 == 0
        amp = 1.2 if hard_zone else 0.6
        freq = 0.3 if hard_zone else 0.15
        phase = rng.random() * math.pi
        events = []
        for t in range(6):
            noise = rng.uniform(-0.25, 0.25) if hard_zone else rng.uniform(-0.05, 0.05)
            value = amp * math.sin(freq * (idx + t) + phase) + 0.1 * t + noise
            state = "hard" if hard_zone else "easy"
            events.append({"timestamp": t, "entity_id": f"u{idx}", "features": {"value": value, "state": state}})
        score = sum(event["features"]["value"] for event in events) / len(events)
        label = 1 if score > 0.45 else 0
        data.append({"sample_id": f"user-{idx}", "events": events, "label": label})
    return data


def summarize_uncertainty_distribution(data: List[Dict[str, Any]], seed: int) -> Dict[str, float]:
    pilot = fp.FlowPilot(random_seed=seed)
    pilot.prepare(data)
    trainer = pilot.trainer
    encoded = pilot.prepared_encoded
    _decision, selected = trainer.select_next_batch(encoded, batch_size=max(1, len(encoded) // 5))
    batch = [item.sample for item in selected]
    trainer.model.train_batch(batch)
    trainer._update_state(batch)
    values = list(trainer.model_state.sample_uncertainty.values())
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
    return {"min": min(values), "max": max(values), "mean": mean, "std": math.sqrt(var)}


def run_flowpilot(data: List[Dict[str, Any]], seed: int, runs: int) -> Dict[str, Any]:
    result = fp.run(data=data, target="classification", baseline_strategy="random", runs=runs, seed=seed)
    print("FlowPilot version:", fp.__version__)
    print("time reduction %:", result.kpi.training_time_reduction_pct)
    print("data efficiency %:", result.kpi.data_efficiency_gain_pct)
    print("step reduction x:", result.kpi.convergence_step_reduction_x)
    print("time reduction ci95:", result.statistics.get("time_reduction_ci95"))
    print("time reduction p-value:", result.statistics.get("time_reduction_p_value"))
    harness = fp.BenchmarkHarness(runs=runs, seed=seed)
    pilot = fp.FlowPilot()
    results = harness.run(pilot=pilot, data=data, epochs=1, batch_size=32)
    baseline_metrics: Dict[str, Dict[str, float]] = {}
    for item in results:
        print("baseline:", item.baseline, "kpi:", item.kpi, "ci95:", item.statistics.get("time_reduction_ci95"))
        baseline_metrics[item.baseline] = item.kpi
    uncertainty_stats = summarize_uncertainty_distribution(data, seed=seed)
    print("uncertainty distribution:", uncertainty_stats)
    return {
        "flowpilot_time_reduction_pct": result.kpi.training_time_reduction_pct,
        "flowpilot_data_efficiency_pct": result.kpi.data_efficiency_gain_pct,
        "flowpilot_step_reduction_x": result.kpi.convergence_step_reduction_x,
        "flowpilot_p_value": result.statistics.get("time_reduction_p_value", 1.0),
        "uncertainty_min": uncertainty_stats["min"],
        "uncertainty_max": uncertainty_stats["max"],
        "uncertainty_mean": uncertainty_stats["mean"],
        "uncertainty_std": uncertainty_stats["std"],
        "baseline_random_time_reduction_pct": baseline_metrics.get("random", {}).get("training_time_reduction_pct", 0.0),
        "baseline_uncertainty_time_reduction_pct": baseline_metrics.get("uncertainty_only", {}).get("training_time_reduction_pct", 0.0),
        "baseline_curriculum_time_reduction_pct": baseline_metrics.get("curriculum_only", {}).get("training_time_reduction_pct", 0.0),
    }


def run_tensorflow_training(data: List[Dict[str, Any]], seed: int, epochs: int) -> Dict[str, Any]:
    try:
        import numpy as np
        import tensorflow as tf
    except Exception as exc:
        print("TensorFlow section skipped. Install with: pip install \".[tensorflow]\"")
        print("Reason:", exc)
        return {"tf_enabled": 0, "tf_confidence": 0.0}

    tf.random.set_seed(seed)

    x = []
    y = []
    for sample in data:
        values = [float(event["features"].get("value", 0.0)) for event in sample["events"]]
        x.append(values[-3:])
        y.append(int(sample["label"]))
    x_train = np.asarray(x, dtype=np.float32)
    y_train = np.asarray(y, dtype=np.float32)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(3,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    trainer = FlowTrainer()
    callback = CurriculumCallback(trainer, monitor="loss", update_every_n_batches=1)
    model.fit(x_train, y_train, epochs=max(1, epochs), batch_size=16, verbose=0, callbacks=[callback])
    state = callback.get_curriculum_state()
    print("TensorFlow fit done")
    print("curriculum strategy:", state["strategy"])
    print("curriculum confidence:", state["avg_confidence"])
    return {
        "tf_enabled": 1,
        "tf_strategy": state["strategy"],
        "tf_confidence": state["avg_confidence"],
    }


def main() -> None:
    size = 360
    runs = 2
    epochs = 2
    seeds = list(range(10))
    records: List[Dict[str, Any]] = []
    for seed in seeds:
        print("===== seed", seed, "=====")
        data = make_flow_data(size=size, seed=seed)
        rec = {"seed": seed}
        rec.update(run_flowpilot(data, seed=seed, runs=runs))
        rec.update(run_tensorflow_training(data, seed=seed, epochs=epochs))
        records.append(rec)

    os.makedirs("results", exist_ok=True)
    output_path = "results/minimal_usage_results.csv"
    fieldnames = sorted({key for row in records for key in row.keys()})
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print("saved:", output_path)


if __name__ == "__main__":
    main()
