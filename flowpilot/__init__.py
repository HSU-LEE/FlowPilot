from __future__ import annotations



from typing import Any, Optional



from flow_engine.flow_benchmark import BenchmarkHarness, BenchmarkResult

from flow_engine.flow_dataloader import FlowBatchPolicy, StepSignal

from flow_engine.flow_encoders import EventLogEncoder, TimeSeriesEncoder

from flow_engine.flow_product import FitResult, FlowPilot



__version__ = "0.1.0"

def _new_pilot(seed: Optional[int] = None) -> FlowPilot:
    if seed is None:
        return FlowPilot()
    return FlowPilot(random_seed=int(seed))





def fit(

    data: list[Any],

    epochs: int = 1,

    batch_size: int = 16,

    mode: str = "auto",

    target: str = "classification",

    baseline_strategy: str = "random",

    runs: int = 3,

    seed: Optional[int] = None,

) -> FitResult:

    """
    Public one-liner entrypoint for pip users.

    Example:
        import flowpilot as fp
        result = fp.fit(data)
    """

    return _new_pilot(seed).fit(

        data=data,

        epochs=epochs,

        batch_size=batch_size,

        mode=mode,

        target=target,

        baseline_strategy=baseline_strategy,

        runs=runs,

        seed=seed,

    )





def run(

    data: list[Any],

    epochs: int = 1,

    batch_size: int = 16,

    mode: str = "auto",

    target: str = "classification",

    baseline_strategy: str = "random",

    runs: int = 3,

    seed: Optional[int] = None,

) -> FitResult:

    """Beginner-friendly one-shot execution API."""

    return fit(

        data=data,

        epochs=epochs,

        batch_size=batch_size,

        mode=mode,

        target=target,

        baseline_strategy=baseline_strategy,

        runs=runs,

        seed=seed,

    )





__all__ = [

    "EventLogEncoder",

    "BenchmarkHarness",

    "BenchmarkResult",

    "FitResult",

    "FlowBatchPolicy",

    "FlowPilot",

    "StepSignal",

    "TimeSeriesEncoder",

    "__version__",

    "fit",

    "run",

]

