"""Single source of truth for world, grid, and angle conversions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class GridFrame:
    origin_x_w: float
    origin_z_w: float
    cell_size_m: float


def normalize_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def world_to_grid(x_w: float, z_w: float, frame: GridFrame) -> tuple[int, int]:
    i = math.floor((x_w - frame.origin_x_w) / frame.cell_size_m)
    j = math.floor((z_w - frame.origin_z_w) / frame.cell_size_m)
    return i, j


def grid_to_world(i: int, j: int, frame: GridFrame) -> tuple[float, float]:
    return (
        frame.origin_x_w + (i + 0.5) * frame.cell_size_m,
        frame.origin_z_w + (j + 0.5) * frame.cell_size_m,
    )


def yaw_from_compass(compass_values: Iterable[float], planar_axes: str = "xz") -> float:
    values = list(compass_values)
    if len(values) < 3:
        raise ValueError("compass_values must contain at least 3 values")
    x_val = float(values[0])
    y_val = float(values[1])
    z_val = float(values[2])
    if planar_axes == "xy":
        return normalize_angle(math.atan2(y_val, x_val))
    if planar_axes == "xz":
        return normalize_angle(math.atan2(x_val, z_val) + math.pi)
    raise ValueError(f"unsupported compass planar axes: {planar_axes}")
