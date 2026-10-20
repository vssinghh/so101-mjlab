"""Tests for SO-ARM101 mjlab robot configuration, tasks, and sim-to-real deployment."""

from pathlib import Path

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import so101_mjlab  # noqa: F401
from so101_mjlab.deploy_real import (
  ACTION_SCALE,
  DEFAULT_JOINT_POS,
  build_state_observation,
)
from so101_mjlab.so101 import get_so101_robot_cfg
from so101_mjlab.sts3215_driver import rad_to_ticks, ticks_to_rad


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


def test_so101_onnx_policy_multi_height_lift() -> None:
  import mujoco
  import onnxruntime as ort

  onnx_path = Path(__file__).resolve().parent.parent / "checkpoints" / "policy.onnx"
  assert onnx_path.exists()
  sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  in_name = sess.get_inputs()[0].name

  robot = Entity(get_so101_robot_cfg())
  mj_model = robot.spec.compile()
  mj_model.opt.timestep = 0.005
  mj_data = mujoco.MjData(mj_model)
  site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")

  for stall_q in (0.175, 0.310):
    for z_init, z_goal in ((0.015, 0.145), (-0.034, 0.115)):
      mj_data.qpos[:6] = DEFAULT_JOINT_POS
      mj_data.qvel[:6] = 0.0
      mujoco.mj_forward(mj_model, mj_data)
      cube_b = np.array([0.235, 0.018, z_init], dtype=np.float32)
      goal_b = np.array([0.235, 0.018, z_goal], dtype=np.float32)
      last_action = np.zeros(6, dtype=np.float32)
      grasped = False
      for _ in range(90):
        q_cur = mj_data.qpos[:6].astype(np.float32).copy()
        dq_cur = mj_data.qvel[:6].astype(np.float32).copy()
        ee_pos = mj_data.site_xpos[site_id].astype(np.float32).copy()
        if (
          not grasped
          and np.linalg.norm(ee_pos - cube_b) < 0.025
          and q_cur[5] < stall_q + 0.08
        ):
          grasped = True
        if grasped:
          cube_b = ee_pos.copy()
        obs = build_state_observation(
          mj_model, mj_data, site_id, q_cur, dq_cur, cube_b, goal_b, last_action
        )
        raw_action = np.asarray(sess.run(None, {in_name: obs})[0]).reshape(-1)[:6]
        mj_data.ctrl[:6] = DEFAULT_JOINT_POS + ACTION_SCALE * raw_action
        mujoco.mj_step(mj_model, mj_data, nstep=4)
        if grasped and mj_data.qpos[5] < stall_q:
          mj_data.qpos[5] = stall_q
          mj_data.qvel[5] = 0.0
          mujoco.mj_forward(mj_model, mj_data)
        last_action = raw_action

      assert grasped, f"Policy failed to grasp at z_init={z_init}, stall_q={stall_q}"
      assert cube_b[2] - z_init > 0.10, (
        f"Policy failed to lift at z_init={z_init}, stall_q={stall_q}"
      )
