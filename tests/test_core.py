from __future__ import annotations
import pytest
import flowpilot as fp

def test_context_get_supports_dotted_keys() -> None:
    ctx = fp.RunContext({'ball': {'position': {'x': 1.0, 'y': 2.0}}})
    assert ctx.get('ball.position.x') == 1.0
    assert ctx.get('ball.position.z', 0.0) == 0.0
    assert 'ball.position.y' in ctx

def test_context_set_is_copy_on_write() -> None:
    ctx = fp.RunContext({'ball': {'position': {'x': 1.0}, 'velocity': [1.0, 0.0]}})
    new_ctx = ctx.set('ball.position.x', 2.0)
    assert ctx.get('ball.position.x') == 1.0
    assert new_ctx.get('ball.position.x') == 2.0
    assert new_ctx.get('ball.velocity') == [1.0, 0.0]

def test_context_set_many_writes_dotted_paths_without_nested_overwrite() -> None:
    ctx = fp.RunContext({'ball': {'position': {'x': 1.0}, 'velocity': [1.0, 0.0]}})
    new_ctx = ctx.set_many({'ball.position.y': 3.0, 'agent.speed': 4.0})
    assert new_ctx.get('ball.position.x') == 1.0
    assert new_ctx.get('ball.position.y') == 3.0
    assert new_ctx.get('ball.velocity') == [1.0, 0.0]
    assert new_ctx.get('agent.speed') == 4.0

def test_context_require_raises_for_missing_path() -> None:
    ctx = fp.RunContext()
    with pytest.raises(KeyError):
        ctx.require('missing.path')

def test_decision_is_frozen_and_mappable() -> None:
    decision = fp.Decision('move', [1, 2])
    mapped = decision.map(tuple)
    assert mapped.kind == 'move'
    assert mapped.value == (1, 2)
