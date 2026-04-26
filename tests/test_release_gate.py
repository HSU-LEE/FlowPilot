from __future__ import annotations

from flow_engine.flow_benchmark import BenchmarkHarness, BenchmarkResult


def test_release_gate_small_runs_uses_non_regression_rule() -> None:
    result = BenchmarkResult(
        baseline="random",
        kpi={
            "training_time_reduction_pct": 0.0,
            "data_efficiency_gain_pct": 0.0,
            "convergence_step_reduction_x": 1.0,
        },
        statistics={"runs": 3, "time_reduction_ci95": [0.0, 0.0]},
    )

    assert BenchmarkHarness.release_gate_passed(result) is True


def test_release_gate_large_runs_requires_positive_ci_lower_bound() -> None:
    failing = BenchmarkResult(
        baseline="random",
        kpi={
            "training_time_reduction_pct": 3.0,
            "data_efficiency_gain_pct": 1.0,
            "convergence_step_reduction_x": 1.1,
        },
        statistics={"runs": 5, "time_reduction_ci95": [0.0, 5.0]},
    )
    passing = BenchmarkResult(
        baseline="random",
        kpi={
            "training_time_reduction_pct": 3.0,
            "data_efficiency_gain_pct": 1.0,
            "convergence_step_reduction_x": 1.1,
        },
        statistics={"runs": 5, "time_reduction_ci95": [0.1, 5.0]},
    )

    assert BenchmarkHarness.release_gate_passed(failing) is False
    assert BenchmarkHarness.release_gate_passed(passing) is True
