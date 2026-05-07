from __future__ import annotations
from typing import Sequence
from .distance import euclidean
__all__ = ['line_intersects_circle', 'point_in_circle', 'point_to_segment_distance']

def point_in_circle(point: Sequence[float], center: Sequence[float], radius: float) -> bool:
    return euclidean(point, center) <= radius

def point_to_segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return euclidean(point, start)
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denom
    t = max(0.0, min(1.0, t))
    proj = (start[0] + t * dx, start[1] + t * dy)
    return euclidean(point, proj)

def line_intersects_circle(start: Sequence[float], end: Sequence[float], center: Sequence[float], radius: float) -> bool:
    return point_to_segment_distance(center, start, end) <= radius
