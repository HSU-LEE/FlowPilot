from __future__ import annotations
import flowpilot as fp

def test_tick_cache_is_runtime_mutable_state() -> None:
    cache = fp.TickCache()
    calls = {'n': 0}

    def compute():
        calls['n'] += 1
        return 7
    assert cache.get_or_compute('x', compute) == 7
    assert cache.get_or_compute('x', compute) == 7
    assert calls['n'] == 1
    cache.reset()
    assert cache.get_or_compute('x', compute) == 7
    assert calls['n'] == 2

def test_clock_steps_and_resets() -> None:
    clock = fp.Clock(dt=0.5)
    clock.step().step()
    assert clock.tick == 2
    assert clock.time == 1.0
    clock.reset()
    assert clock.tick == 0
    assert clock.time == 0.0

def test_scheduler_runs_due_tasks_only() -> None:
    scheduler = fp.Scheduler()
    scheduler.add('every_two', 2, lambda ctx: ctx.set('ran', ctx.get('ran', 0) + 1))
    ctx = fp.RunContext()
    ctx = scheduler.run_due(ctx, tick=0)
    ctx = scheduler.run_due(ctx, tick=1)
    ctx = scheduler.run_due(ctx, tick=2)
    assert ctx.get('ran') == 2

def test_tick_loop_wraps_pipeline_and_sets_runtime_keys() -> None:
    node = fp.Node(name='accumulate', inputs=('count', 'runtime.dt'), outputs=('count',), compute=lambda count, dt: count + dt)
    loop = fp.TickLoop(fp.Pipeline((node,)), dt=0.5)
    ctx = loop.run(fp.RunContext({'count': 0.0}), ticks=3)
    assert ctx.get('count') == 1.5
    assert ctx.get('runtime.tick') == 2
    assert loop.clock.tick == 3
