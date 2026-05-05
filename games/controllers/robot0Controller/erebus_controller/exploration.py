"""Frontier scoring helpers."""

from __future__ import annotations

import math


def score_frontier(
    robot_cell: tuple[int, int],
    frontier: tuple[int, int],
    visit_count: int,
    heading_error: float,
) -> float:
    dx = frontier[0] - robot_cell[0]
    dy = frontier[1] - robot_cell[1]
    return math.hypot(dx, dy) + visit_count * 0.35 + abs(heading_error) * 0.50
