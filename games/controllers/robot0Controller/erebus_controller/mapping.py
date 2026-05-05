"""Occupancy grid mapping, inflation, terrain costs, and frontier extraction."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .utils.frames import GridFrame, grid_to_world, world_to_grid

UNKNOWN = -1
FREE = 0
OCCUPIED = 1
HAZARD = 2


@dataclass(frozen=True)
class OccupancyConfig:
    width_cells: int
    height_cells: int
    inflation_radius_cells: int = 1


class OccupancyGridMap:
    def __init__(self, frame: GridFrame, config: OccupancyConfig) -> None:
        self.frame = frame
        self.config = config
        self.grid = [[UNKNOWN for _ in range(config.width_cells)] for _ in range(config.height_cells)]
        self._inflated = [[False for _ in range(config.width_cells)] for _ in range(config.height_cells)]
        self._visited: dict[tuple[int, int], int] = {}
        self._terrain_penalty: dict[tuple[int, int], float] = {}
        self._terrain_kind: dict[tuple[int, int], str] = {}

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.config.width_cells and 0 <= j < self.config.height_cells

    def world_to_grid(self, x_w: float, z_w: float) -> tuple[int, int]:
        return world_to_grid(x_w, z_w, self.frame)

    def grid_to_world(self, i: int, j: int) -> tuple[float, float]:
        return grid_to_world(i, j, self.frame)

    def mark_free(self, i: int, j: int) -> None:
        if not self.in_bounds(i, j):
            return
        if self.grid[j][i] in (OCCUPIED, HAZARD):
            return
        self.grid[j][i] = FREE

    def mark_free_world(self, x_w: float, z_w: float) -> None:
        self.mark_free(*self.world_to_grid(x_w, z_w))

    def mark_occupied(self, i: int, j: int) -> None:
        if not self.in_bounds(i, j) or self.grid[j][i] == HAZARD:
            return
        self.grid[j][i] = OCCUPIED
        self._rebuild_inflation()

    def mark_occupied_world(self, x_w: float, z_w: float) -> None:
        self.mark_occupied(*self.world_to_grid(x_w, z_w))

    def mark_hazard(self, i: int, j: int, kind: str = "hazard") -> None:
        if not self.in_bounds(i, j):
            return
        self.grid[j][i] = HAZARD
        self._terrain_kind[(i, j)] = kind
        self._rebuild_inflation()

    def mark_hazard_world(self, x_w: float, z_w: float, kind: str = "hazard") -> None:
        self.mark_hazard(*self.world_to_grid(x_w, z_w), kind=kind)

    def mark_visited_world(self, x_w: float, z_w: float) -> None:
        cell = self.world_to_grid(x_w, z_w)
        if not self.in_bounds(*cell):
            return
        self._visited[cell] = self._visited.get(cell, 0) + 1
        self.mark_free(*cell)

    def visit_count(self, i: int, j: int) -> int:
        return self._visited.get((i, j), 0)

    def mark_zone_world(
        self,
        x_min: float,
        x_max: float,
        z_min: float,
        z_max: float,
        kind: str,
        penalty_cost: float,
    ) -> None:
        for j in range(self.config.height_cells):
            for i in range(self.config.width_cells):
                x_w, z_w = self.grid_to_world(i, j)
                half = self.frame.cell_size_m * 0.5
                if x_w + half < x_min or x_w - half > x_max or z_w + half < z_min or z_w - half > z_max:
                    continue
                self._terrain_kind[(i, j)] = kind
                if kind in ("trap", "hazard"):
                    self.grid[j][i] = HAZARD
                elif penalty_cost > 0.0:
                    self._terrain_penalty[(i, j)] = max(self._terrain_penalty.get((i, j), 0.0), penalty_cost)
        self._rebuild_inflation()

    def update_from_scan(
        self,
        robot_x_w: float,
        robot_z_w: float,
        robot_yaw: float,
        rays: list[tuple[float, float]],
        max_range_m: float,
    ) -> None:
        start = self.world_to_grid(robot_x_w, robot_z_w)
        if not self.in_bounds(*start):
            return
        self.mark_free(*start)
        for rel_yaw, raw_distance in rays:
            distance = max(0.0, min(max_range_m, raw_distance))
            beam_yaw = robot_yaw + rel_yaw
            end_x = robot_x_w + math.cos(beam_yaw) * distance
            end_z = robot_z_w + math.sin(beam_yaw) * distance
            end = self.world_to_grid(end_x, end_z)
            cells = self._bresenham(start, end)
            if not cells:
                continue
            hit_obstacle = distance < max_range_m * 0.94
            free_cells = cells[:-1] if hit_obstacle else cells
            for cell in free_cells:
                self.mark_free(*cell)
            if hit_obstacle:
                self.mark_occupied(*cells[-1])

    def mark_obstacle_ahead(self, x_w: float, z_w: float, yaw: float, distance: float) -> None:
        self.mark_occupied_world(x_w + math.cos(yaw) * distance, z_w + math.sin(yaw) * distance)

    def _bresenham(self, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        x0, y0 = start
        x1, y1 = end
        cells: list[tuple[int, int]] = []
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                return cells
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def _rebuild_inflation(self) -> None:
        width = self.config.width_cells
        height = self.config.height_cells
        self._inflated = [[False for _ in range(width)] for _ in range(height)]
        radius = max(0, self.config.inflation_radius_cells)
        for j in range(height):
            for i in range(width):
                if self.grid[j][i] not in (OCCUPIED, HAZARD):
                    continue
                for dj in range(-radius, radius + 1):
                    for di in range(-radius, radius + 1):
                        ni = i + di
                        nj = j + dj
                        if self.in_bounds(ni, nj):
                            self._inflated[nj][ni] = True

    def is_traversable(self, i: int, j: int) -> bool:
        if not self.in_bounds(i, j):
            return False
        if self.grid[j][i] != FREE:
            return False
        return not self._inflated[j][i]

    def is_known_safe(self, i: int, j: int) -> bool:
        return self.in_bounds(i, j) and self.grid[j][i] == FREE

    def cell_cost(self, i: int, j: int) -> float:
        if not self.in_bounds(i, j) or self.grid[j][i] == HAZARD:
            return float("inf")
        cost = self._terrain_penalty.get((i, j), 0.0)
        if self._inflated[j][i]:
            cost += 2.0
        cost += min(4.0, self.visit_count(i, j) * 0.08)
        return cost

    def nearest_traversable(self, target: tuple[int, int], max_radius: int = 6) -> tuple[int, int] | None:
        if self.is_traversable(*target):
            return target
        ti, tj = target
        best: tuple[float, tuple[int, int]] | None = None
        for radius in range(1, max_radius + 1):
            for dj in range(-radius, radius + 1):
                for di in range(-radius, radius + 1):
                    if max(abs(di), abs(dj)) != radius:
                        continue
                    cell = (ti + di, tj + dj)
                    if not self.is_traversable(*cell):
                        continue
                    dist = math.hypot(di, dj)
                    if best is None or dist < best[0]:
                        best = (dist, cell)
            if best is not None:
                return best[1]
        return None

    def get_frontiers(self) -> list[tuple[int, int]]:
        frontiers: list[tuple[int, int]] = []
        for j in range(self.config.height_cells):
            for i in range(self.config.width_cells):
                if not self.is_traversable(i, j):
                    continue
                if self._adjacent_unknown(i, j):
                    frontiers.append((i, j))
        return frontiers

    def _adjacent_unknown(self, i: int, j: int) -> bool:
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni = i + di
            nj = j + dj
            if self.in_bounds(ni, nj) and self.grid[nj][ni] == UNKNOWN:
                return True
        return False

    def explored_ratio(self) -> float:
        total = self.config.width_cells * self.config.height_cells
        known = sum(1 for row in self.grid for value in row if value != UNKNOWN)
        return 0.0 if total == 0 else known / float(total)

    def to_text_rows(self) -> list[str]:
        chars = {UNKNOWN: "?", FREE: ".", OCCUPIED: "#", HAZARD: "!"}
        return ["".join(chars.get(value, "?") for value in row) for row in reversed(self.grid)]
