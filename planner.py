"""SICNav (bilevel-MPC) wrapper for the arena_planners bridge.

CollisionAvoidMPC is a pure CasADi/IPOPT bilevel MPC (no learned weights). It
imports matplotlib and rvo2 at module load, so MPLBACKEND must be set to Agg and
Python-RVO2 must be built into the venv (see CMakeLists.txt). Vendored upstream
subtrees (sicnav/, crowd_sim_plus/) live alongside this file and are exposed via
sys.path before any sicnav import.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import configparser
import pathlib
import sys

import numpy as np
from arena_planners.sdk import load_manifest, main_loop

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))

from crowd_sim_plus.envs.utils.human_plus import Human  # noqa: E402
from crowd_sim_plus.envs.utils.state_plus import (  # noqa: E402
    FullState,
    JointState,
    ObservableState,
)
from sicnav.policy.campc import CollisionAvoidMPC  # noqa: E402

_TIME_STEP: float = 0.25
_ROBOT_RADIUS: float = 0.3
_HUMAN_RADIUS: float = 0.35
_V_PREF: float = 0.9
_MAX_SPEED: float = 0.95
_MAX_NUM_HUMS: int = 2

_policy: CollisionAvoidMPC | None = None
_built_num_hums: int | None = None


class _DummyEnv:
    """Minimal stand-in for the gym env that CollisionAvoidMPC.set_env reads.

    set_env consults env.config (a RawConfigParser), env.global_time, and
    env.set_human_observability. No gym machinery is exercised at inference.
    """

    def __init__(self, config: configparser.RawConfigParser) -> None:
        self.config = config
        self.global_time = 0.0
        self.time_step = _TIME_STEP

    def set_human_observability(self, _observable: bool) -> None:
        return


def _load_config() -> configparser.RawConfigParser:
    config = configparser.RawConfigParser()
    config.read(str(_HERE / "configs" / "policy.config"))
    config.read(str(_HERE / "configs" / "env.config"))
    return config


def _build_policy(config: configparser.RawConfigParser) -> CollisionAvoidMPC:
    policy = CollisionAvoidMPC()
    policy.configure(config)
    policy.time_step = _TIME_STEP
    policy.set_phase("test")
    policy.set_device("cpu")
    dummy_human = Human(config, section="humans", fully_observable=False)
    policy.dummy_human = dummy_human
    policy.env = _DummyEnv(config)
    return policy


def _make_joint_state(features: dict) -> JointState | None:
    robot_pose = features.get("robot_pose")
    robot_state = features.get("robot_state")
    goal_pose = features.get("goal_pose")
    if robot_pose is None or robot_state is None or goal_pose is None:
        return None
    if len(robot_pose) < 3 or len(goal_pose) < 2:
        return None

    px, py, theta = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    vx, vy = float(robot_state[2]), float(robot_state[3])
    gx, gy = float(goal_pose[0]), float(goal_pose[1])

    self_state = FullState(px, py, vx, vy, _ROBOT_RADIUS, gx, gy, _V_PREF, theta)

    humans: list[tuple[float, ObservableState]] = []
    _peds = features.get("pedestrians")
    for ped in (() if _peds is None else _peds):
        hpx, hpy = float(ped[1]), float(ped[2])
        hvx, hvy = float(ped[3]), float(ped[4])
        dist = float(np.hypot(hpx - px, hpy - py))
        humans.append((dist, ObservableState(hpx, hpy, hvx, hvy, _HUMAN_RADIUS)))
    humans.sort(key=lambda t: t[0])
    human_states = [h for _, h in humans[:_MAX_NUM_HUMS]]

    return JointState(
        self_state=self_state,
        human_states=human_states,
        static_obs=[],
    )


def step(features: dict) -> list[float]:
    """Map features to a SICNav bilevel-MPC action, return [v, omega]."""
    global _policy, _built_num_hums
    if _policy is None:
        _policy = _build_policy(_load_config())

    joint_state = _make_joint_state(features)
    if joint_state is None:
        return [0.0, 0.0]

    num_hums = len(joint_state.human_states)
    if not num_hums:
        sx = joint_state.self_state
        dx, dy = sx.gx - sx.px, sx.gy - sx.py
        dist = float(np.hypot(dx, dy))
        if dist < 1e-6:
            return [0.0, 0.0]
        v = min(_V_PREF, dist)
        omega = float(np.arctan2(dy, dx) - sx.theta)
        omega = (omega + np.pi) % (2 * np.pi) - np.pi
        return [v, max(-1.0, min(1.0, omega))]

    if _built_num_hums != num_hums:
        _policy.num_hums = num_hums
        _policy.mpc_env = None
        _policy.env.global_time = 0.0
        _built_num_hums = num_hums
        return [0.0, 0.0]

    action = _policy.predict(joint_state)
    omega = float(action.r) / _TIME_STEP
    v = float(action.v)
    if not (np.isfinite(v) and np.isfinite(omega)):
        return [0.0, 0.0]
    return [max(0.0, min(_MAX_SPEED, v)), omega]


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    global _policy, _built_num_hums
    if _policy is not None:
        _policy.mpc_env = None
        _policy.env.global_time = 0.0
    _built_num_hums = None


if __name__ == "__main__":
    manifest = load_manifest(_HERE / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset)
