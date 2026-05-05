"""Deterministic grid A* planner."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Callable

GridCell = tuple[int, int]


@dataclass(frozen=True)
class AStarConfig:
    allow_diagonal: bool = True


class AStarPlanner:
    def __init__(self, config: AStarConfig | None = None) -> None:
        self.config = config or AStarConfig()

    def plan(
        self,
        start: GridCell,
        goal: GridCell,
        is_traversable: Callable[[int, int], bool],
        traversal_cost: Callable[[int, int], float] | None = None,
    ) -> list[GridCell] | None:
        if not is_traversable(*start) or not is_traversable(*goal):
            return None
        if start == goal:
            return [start]

        traversal_cost = traversal_cost or (lambda _i, _j: 0.0)
        frontier: list[tuple[float, int, GridCell]] = []
        came_from: dict[GridCell, GridCell] = {}
        g_score: dict[GridCell, float] = {start: 0.0}
        visited: set[GridCell] = set()
        order = 0
        heapq.heappush(frontier, (self._heuristic(start, goal), order, start))

        while frontier:
            _prio, _order, current = heapq.heappop(frontier)
            if current in visited:
                continue
            if current == goal:
                return self._reconstruct(came_from, current)
            visited.add(current)

            for neighbor, step_cost in self._neighbors(current):
                if not is_traversable(*neighbor):
                    continue
                if step_cost > 1.0 and not self._diagonal_allowed(current, neighbor, is_traversable):
                    continue
                tentative = g_score[current] + step_cost + max(0.0, traversal_cost(*neighbor))
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                order += 1
                heapq.heappush(frontier, (tentative + self._heuristic(neighbor, goal), order, neighbor))
        return None

    def _neighbors(self, cell: GridCell) -> list[tuple[GridCell, float]]:
        i, j = cell
        result = [((i + 1, j), 1.0), ((i - 1, j), 1.0), ((i, j + 1), 1.0), ((i, j - 1), 1.0)]
        if self.config.allow_diagonal:
            root2 = math.sqrt(2.0)
            result.extend(
                [
                    ((i + 1, j + 1), root2),
                    ((i + 1, j - 1), root2),
                    ((i - 1, j + 1), root2),
                    ((i - 1, j - 1), root2),
                ]
            )
        return result

    def _heuristic(self, a: GridCell, b: GridCell) -> float:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if not self.config.allow_diagonal:
            return dx + dy
        return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)

    @staticmethod
    def _diagonal_allowed(
        current: GridCell,
        neighbor: GridCell,
        is_traversable: Callable[[int, int], bool],
    ) -> bool:
        di = neighbor[0] - current[0]
        dj = neighbor[1] - current[1]
        return is_traversable(current[0] + di, current[1]) and is_traversable(current[0], current[1] + dj)

    @staticmethod
    def _reconstruct(came_from: dict[GridCell, GridCell], current: GridCell) -> list[GridCell]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
