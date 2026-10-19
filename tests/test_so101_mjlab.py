"""Tests for SO-ARM101 mjlab robot configuration, tasks, and sim-to-real deployment."""

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import so101_mjlab  # noqa: F401
from so101_mjlab.deploy_real import (
  DEFAULT_JOINT_POS,
  build_state_observation,
  rad_to_ticks,
  ticks_to_rad,
)
from so101_mjlab.so101 import get_so101_robot_cfg


def test_so101_entity_compiles() -> None:
  robot = Entity(get_so101_robot_cfg())
  model = robot.spec.compile()
  assert model.nq == 6
  assert model.nv == 6
  assert model.nu == 6
  assert model.ncam == 1


def test_so101_tasks_registered() -> None:
  tasks = list_tasks()
  for task_id in (
    "Mjlab-Lift-Cube-SO101",
    "Mjlab-Lift-Cube-SO101-Rgb",
    "Mjlab-Lift-Cube-SO101-Depth",
  ):
    assert task_id in tasks
    env_cfg = load_env_cfg(task_id)
    rl_cfg = load_rl_cfg(task_id)
    assert env_cfg is not None
    assert rl_cfg is not None


def test_so101_lift_cube_env_step() -> None:
  cfg = load_env_cfg("Mjlab-Lift-Cube-SO101", play=True)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  obs, _ = env.reset()
  assert obs["actor"].shape == (2, 24)

  actions = torch.zeros((2, 6), dtype=torch.float32)
  obs, rew, terminated, truncated, _ = env.step(actions)
  assert obs["actor"].shape == (2, 24)
  assert rew.shape == (2,)
  assert terminated.shape == (2,)
  assert truncated.shape == (2,)
  env.close()


def test_so101_calibration_roundtrip_and_fk_obs() -> None:
  ticks = rad_to_ticks(DEFAULT_JOINT_POS)
  q_recovered = ticks_to_rad(ticks)
  np.testing.assert_allclose(q_recovered, DEFAULT_JOINT_POS, atol=2e-3)

  robot = Entity(get_so101_robot_cfg())
  mj_model = robot.spec.compile()
  import mujoco

  mj_data = mujoco.MjData(mj_model)
  site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
  obs = build_state_observation(
    mj_model=mj_model,
    mj_data=mj_data,
    site_id=site_id,
    q_rad=DEFAULT_JOINT_POS,
    dq_rad=np.zeros(6, dtype=np.float32),
    cube_pos_b=np.array([0.22, 0.0, 0.015], dtype=np.float32),
    goal_pos_b=np.array([0.22, 0.0, 0.16], dtype=np.float32),
    last_action=np.zeros(6, dtype=np.float32),
  )
  assert obs.shape == (1, 24)
