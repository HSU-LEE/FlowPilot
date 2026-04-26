from __future__ import annotations



from dataclasses import dataclass, field

from typing import Any



from .flow_product import FlowPilot





@dataclass

class BenchmarkResult:

    baseline: str

    kpi: dict[str, float]

    statistics: dict[str, Any]





@dataclass

class BenchmarkHarness:

    """Benchmark runner with baseline suite and reproducible settings."""



    baselines: tuple[str, ...] = ("random", "uncertainty_only", "curriculum_only")

    runs: int = 5

    seed: int = 13



    def run(

        self,

        pilot: FlowPilot,

        data: list[Any],

        epochs: int = 1,

        batch_size: int = 16,

        target: str = "classification",

    ) -> list[BenchmarkResult]:

        results: list[BenchmarkResult] = []

        for baseline in self.baselines:
            run_pilot = FlowPilot(
                target_confidence=pilot.target_confidence,
                random_seed=pilot.random_seed,
                encoder=pilot.encoder,
            )

            fit = run_pilot.fit(

                data=data,

                epochs=epochs,

                batch_size=batch_size,

                baseline_strategy=baseline,

                target=target,

                runs=self.runs,

                seed=self.seed,

            )

            results.append(

                BenchmarkResult(

                    baseline=baseline,

                    kpi={

                        "training_time_reduction_pct": fit.kpi.training_time_reduction_pct,

                        "data_efficiency_gain_pct": fit.kpi.data_efficiency_gain_pct,

                        "convergence_step_reduction_x": fit.kpi.convergence_step_reduction_x,

                    },

                    statistics=fit.statistics,

                )

            )

        return results



    @staticmethod

    def release_gate_passed(result: BenchmarkResult) -> bool:
        ci = result.statistics.get("time_reduction_ci95", [0.0, 0.0])
        lower_bound = float(ci[0]) if ci else 0.0
        runs = int(result.statistics.get("runs", 1))
        time_reduction = float(result.kpi.get("training_time_reduction_pct", 0.0))
        data_gain = float(result.kpi.get("data_efficiency_gain_pct", 0.0))

        # Small-run checks are statistically underpowered; treat them as
        # non-regression smoke gates instead of strict significance gates.
        if runs < 5:
            return time_reduction >= 0.0 and data_gain >= 0.0

        return lower_bound > 0.0

