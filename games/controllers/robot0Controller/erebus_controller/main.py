"""Webots entrypoint for the Erebus controller."""

from __future__ import annotations

from .agent import ErebusAgent
from .robot_io import RobotIO


def run() -> None:
    try:
        from controller import Robot
    except ImportError as exc:
        raise RuntimeError("Run this controller inside Webots (module 'controller' is required).") from exc

    robot = Robot()
    io = RobotIO(robot)
    agent = ErebusAgent(io)
    try:
        while io.step():
            agent.tick()
    finally:
        io.set_wheel_speeds(0.0, 0.0)


if __name__ == "__main__":
    run()

