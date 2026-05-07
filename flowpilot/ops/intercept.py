from __future__ import annotations
import math
from typing import Sequence
__all__ = ['intercept_point', 'time_to_intercept']

def time_to_intercept(pursuer_pos: Sequence[float], pursuer_speed: float, target_pos: Sequence[float], target_vel: Sequence[float]) -> float | None:
    if pursuer_speed <= 0:
        return None
    rx = target_pos[0] - pursuer_pos[0]
    ry = target_pos[1] - pursuer_pos[1]
    vx = target_vel[0]
    vy = target_vel[1]
    a = vx * vx + vy * vy - pursuer_speed * pursuer_speed
    b = 2.0 * (rx * vx + ry * vy)
    c = rx * rx + ry * ry
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return 0.0 if c <= 1e-12 else None
        t = -c / b
        return t if t >= 0 else None
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None
    root = math.sqrt(disc)
    candidates = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
    valid = [t for t in candidates if t >= 0]
    return min(valid) if valid else None

def intercept_point(pursuer_pos: Sequence[float], pursuer_speed: float, target_pos: Sequence[float], target_vel: Sequence[float]) -> list[float] | None:
    t = time_to_intercept(pursuer_pos, pursuer_speed, target_pos, target_vel)
    if t is None:
        return None
    return [target_pos[i] + target_vel[i] * t for i in range(len(target_pos))]
