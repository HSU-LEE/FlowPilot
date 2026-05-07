# FlowPilot

A small execution framework for tick-loop decision systems.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Tests](https://img.shields.io/badge/tests-24%20passing-brightgreen)]()
[![Deps](https://img.shields.io/badge/dependencies-zero-lightgrey)]()

```bash
pip install -e ".[dev]"
```

---

## Overview

Game bots, simulation controllers, robotics loops — anything that runs
on a clock and asks *"what next?"* tends to end up with the same
shape: gather observations, predict, score, pick. FlowPilot pins that
shape down with two pieces: **Node** and **Pipeline**. It is not a
neural-net library. It is the layer above your model — learned policy
or hand-tuned rules — that organises the per-tick decision flow.

Pure Python, zero runtime dependencies.

## Three core contracts

```python
import flowpilot as fp

ctx  = fp.RunContext({"x": 1.0})            # dotted keys + copy-on-write
node = fp.Node(                              # pure op with declared I/O
    name="square",
    inputs=("x",), outputs=("y",),
    compute=lambda x: x * x,
)
pipe = node >> node                          # define graph (lazy)
pipe.run(ctx).get("y")                       # execution only on .run()
```

- `RunContext` — an immutable-ish key/value store with dotted-path
  access. Every write returns a new object; `ctx.set("a.b.c", v)`
  does not damage sibling keys like `a.b.d`.
- `Node` — a frozen dataclass declaring `name / inputs / outputs /
  compute`. `forward(ctx)` reads the input keys, calls `compute`,
  validates the output shape, and returns a new context. Mismatches
  raise `NodeOutputError`.
- `Pipeline` — a sequential DAG built by chaining nodes with `>>`.
  Execution only happens when `.run(ctx)` is called.

## Quickstart

```python
import flowpilot as fp
from flowpilot.ops import distance

observe = fp.Node("observe", ("raw",),
                  ("self.pos", "target.pos"),
                  lambda r: (r["me"], r["target"]))

score   = fp.Node("score", ("self.pos", "target.pos"),
                  ("score",),
                  lambda a, b: -distance.euclidean(a, b))

decide  = fp.Node("decide", ("score",),
                  ("decision",),
                  lambda s: fp.Decision("approach", s))

agent = observe >> score >> decide

ctx = agent.run(fp.RunContext({"raw": {"me": (0, 0), "target": (3, 4)}}))
ctx.get("decision")     # Decision(kind='approach', value=-5.0)
```

When you need time to pass, wrap the pipeline in a `TickLoop`. It
injects `runtime.tick / time / dt` into the context every tick, and
its `Scheduler` lets sub-pipelines run at different rates.

```python
loop = fp.TickLoop(agent, dt=0.05)
loop.scheduler.add("planner", every=10, step=replan.run)
ctx  = loop.run(fp.RunContext({"raw": obs}), ticks=240)
```

## Module layout

```
flowpilot/
├── core/      RunContext, Decision, Block
├── ops/       distance, angle, normalize, scoring, collision, intercept
├── graph/     Node, Pipeline, NodeOutputError
└── runtime/   TickCache, Clock, Scheduler, TickLoop
```

`ops/` contains stateless pure functions only. Mutable state is
allowed exclusively inside `runtime/`.

## Design rules

- **`compute` is a pure function.** State belongs in `runtime/`.
- **Immutable dataflow:** `ctx → forward(ctx) → new_ctx`.
- **Explicit contracts.** Inputs and outputs are declared and checked
  at runtime.
- **Sequential DAG only.** Branching, parallelism, and graph
  optimisation are deferred until they are actually needed.
- **README before code.** How users experience the framework matters
  more than its internals.

## Install · Test

```bash
pip install -e ".[dev]"
pytest -q                       # 24 tests: core / ops / graph / runtime / integration
```

## License

MIT.
