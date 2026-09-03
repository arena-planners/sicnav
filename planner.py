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
    FullyObservableJointState,
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
_DUMMY_HUMAN_DIST: float = 50.0
_DUMMY_HUMAN_SPACING: float = 5.0
_WARMUP_GOAL_DIST: float = 5.0

_policy: CollisionAvoidMPC | None = None
_pending_scenario_reset: bool = True


class _DummyEnv:
    """Minimal stand-in for the gym env that CollisionAvoidMPC.set_env reads.

    set_env consults env.config (a RawConfigParser), env.global_time, and
    env.set_human_observability. No gym machinery is exercised at inference.
    """

    def __init__(self, config: configparser.RawConfigParser) -> None:
        self.config = config
        self.global_time = 0.0
        self.time_step = _TIME_STEP
        self.sim_env = ""

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


def _dummy_human_state(px: float, py: float, gx: float, gy: float, slot: int) -> ObservableState:
    dx, dy = px - gx, py - gy
    dist = float(np.hypot(dx, dy))
    ux, uy = (1.0, 0.0) if dist < 1e-6 else (dx / dist, dy / dist)
    lateral = _DUMMY_HUMAN_SPACING * slot
    return ObservableState(
        px + ux * _DUMMY_HUMAN_DIST - uy * lateral,
        py + uy * _DUMMY_HUMAN_DIST + ux * lateral,
        0.0,
        0.0,
        _HUMAN_RADIUS,
    )


def _pad_human_states(
    human_states: list[ObservableState], px: float, py: float, gx: float, gy: float
) -> list[ObservableState]:
    padded = list(human_states[:_MAX_NUM_HUMS])
    while len(padded) < _MAX_NUM_HUMS:
        padded.append(_dummy_human_state(px, py, gx, gy, len(padded)))
    return padded


def _warm_up_state() -> JointState:
    self_state = FullState(0.0, 0.0, 0.0, 0.0, _ROBOT_RADIUS, _WARMUP_GOAL_DIST, 0.0, _V_PREF, 0.0)
    human_states = _pad_human_states([], 0.0, 0.0, _WARMUP_GOAL_DIST, 0.0)
    return JointState(self_state=self_state, human_states=human_states, static_obs=[])


def _human_tracking_state(policy: CollisionAvoidMPC, joint_state: JointState) -> JointState:
    """Reproduce campc.predict's non-privileged state wrap, for reset_humans calls."""
    if policy.priviledged_info:
        return joint_state
    human_states = [
        FullState(
            px=hum.px,
            py=hum.py,
            vx=hum.vx,
            vy=hum.vy,
            gx=hum.px + hum.vx * 2,
            gy=hum.py + hum.vy * 2,
            v_pref=policy.human_max_speed,
            theta=float(np.arctan2(hum.vy, hum.vx)),
            radius=hum.radius,
        )
        for hum in joint_state.human_states
    ]
    return FullyObservableJointState(
        self_state=joint_state.self_state,
        human_states=human_states,
        static_obs=joint_state.static_obs,
    )


def _ensure_policy() -> CollisionAvoidMPC:
    global _policy
    if _policy is None:
        policy = _build_policy(_load_config())
        policy.predict(_warm_up_state())
        policy.env.global_time = 1.0
        policy.reset_scenario_values()
        _policy = policy
    return _policy


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
    human_states = _pad_human_states([h for _, h in humans[:_MAX_NUM_HUMS]], px, py, gx, gy)

    return JointState(
        self_state=self_state,
        human_states=human_states,
        static_obs=[],
    )


def step(features: dict) -> list[float]:
    """Map features to a SICNav bilevel-MPC action, return [v, omega]."""
    global _pending_scenario_reset
    policy = _ensure_policy()

    joint_state = _make_joint_state(features)
    if joint_state is None:
        return [0.0, 0.0]

    if _pending_scenario_reset:
        tracking_state = _human_tracking_state(policy, joint_state)
        policy.mpc_env.callback_orca.reset_humans(tracking_state)
        policy.mpc_env.casadi_orca.reset_humans(tracking_state)
        policy.reset_scenario_values()
        _pending_scenario_reset = False

    policy.gen_ref_traj(_human_tracking_state(policy, joint_state))
    action = policy.predict(joint_state)
    omega = float(action.r) / _TIME_STEP
    v = float(action.v)
    if not (np.isfinite(v) and np.isfinite(omega)):
        return [0.0, 0.0]
    return [max(0.0, min(_MAX_SPEED, v)), omega]


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    global _pending_scenario_reset
    _pending_scenario_reset = True


if __name__ == "__main__":
    manifest = load_manifest(_HERE / "planner.yaml")
    _ensure_policy()
    main_loop(step, manifest=manifest, on_reset=on_reset)
