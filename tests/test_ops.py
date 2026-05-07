from __future__ import annotations
import math
import pytest
from flowpilot.ops import angle, collision, distance, intercept, normalize, scoring

def test_distance_ops() -> None:
    assert distance.euclidean([0, 0], [3, 4]) == 5
    assert distance.squared([0, 0], [3, 4]) == 25
    assert distance.manhattan([0, 0], [3, 4]) == 7
    assert distance.chebyshev([0, 0], [3, 4]) == 4

def test_angle_ops() -> None:
    assert math.isclose(angle.wrap_to_pi(math.pi * 3), -math.pi)
    assert math.isclose(angle.bearing_to([0, 0], [0, 1]), math.pi / 2)
    assert angle.bearing_within([0, 0], 0.0, [1, 0], 0.1)

def test_normalize_ops() -> None:
    assert normalize.l2_normalize([3, 4]) == [0.6, 0.8]
    assert normalize.l2_normalize([0, 0]) == [0, 0]
    assert normalize.scale_to_range(5, 0, 10, -1, 1) == 0
    assert normalize.clamp(2, 0, 1) == 1
    with pytest.raises(ValueError):
        normalize.clamp(0, 1, 0)

def test_scoring_ops() -> None:
    assert scoring.weighted_sum([1, 2, 3], [4, 5, 6]) == 32
    probs = scoring.softmax([1.0, 2.0])
    assert math.isclose(sum(probs), 1.0)
    assert scoring.top_k([1, 3, 2], 2) == [3, 2]
    with pytest.raises(ValueError):
        scoring.weighted_sum([1], [1, 2])

def test_collision_ops() -> None:
    assert collision.point_in_circle([1, 0], [0, 0], 1)
    assert collision.line_intersects_circle([0, 0], [4, 0], [2, 0.5], 0.6)
    assert math.isclose(collision.point_to_segment_distance([2, 1], [0, 0], [4, 0]), 1.0)

def test_intercept_ops() -> None:
    t = intercept.time_to_intercept([0, 0], 2.0, [4, 0], [0, 0])
    assert t == 2.0
    p = intercept.intercept_point([0, 0], 2.0, [4, 0], [1, 0])
    assert p == [8.0, 0.0]
    assert intercept.time_to_intercept([0, 0], 0.0, [1, 0], [0, 0]) is None
