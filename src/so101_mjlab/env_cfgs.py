"""SO-ARM101 manipulation environment configurations for mjlab."""

import math
from typing import TYPE_CHECKING, Any, Literal, cast

import mujoco
import torch
from mjlab.entity import Entity, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommand, LiftingCommandCfg

from so101_mjlab.so101.so101_constants import (
  SO101_ACTION_SCALE,
  get_so101_robot_cfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def so101_grasp_and_lift_reward(
  env: "ManagerBasedRlEnv",
  command_name: str,
  object_name: str,
  pad_cfg: SceneEntityCfg,
  gripper_cfg: SceneEntityCfg,
  bringing_std: float = 0.18,
) -> torch.Tensor:
  """Dense two-jaw grasp and lift reward tailored to the SO-101 rotary jaw gripper."""
  robot: Entity = env.scene[pad_cfg.name]
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))

  obj_pos_w = obj.data.root_link_pos_w
  pad_pos_w = robot.data.geom_pos_w[:, pad_cfg.geom_ids]
  p_fixed = pad_pos_w[:, 0]
  p_moving = pad_pos_w[:, 1]
  p_mid = 0.5 * (p_fixed + p_moving)

  d_mid = torch.norm(p_mid - obj_pos_w, dim=-1)
  d_fixed = torch.norm(p_fixed - obj_pos_w, dim=-1)
  d_moving = torch.norm(p_moving - obj_pos_w, dim=-1)

  q_grip = robot.data.joint_pos[:, gripper_cfg.joint_ids].squeeze(-1)
  q_roll = robot.data.joint_pos[:, 4]
  act_grip = env.action_manager.action[:, -1]

  # Keep the wrist roll horizontal (-pi/2 rad) so the jaws stay level with the table.
  level_wrist = torch.exp(-torch.square((q_roll + 1.5708) / 0.25))

  # Midpoint of fixed_jaw_pad and moving_jaw_pad only reaches the cube center
  # when the two pads straddle opposite faces of the cube.
  in_jaws = torch.exp(-torch.square(d_mid / 0.018))

  # Encourage opening the jaws wide (q_grip >= 0.55 rad) while approaching the object.
  open_approach = (1.0 - in_jaws) * torch.sigmoid((q_grip - 0.52) / 0.04)

  err_fixed = (d_fixed - 0.025).clamp_min(0.0)
  err_moving = (d_moving - 0.025).clamp_min(0.0)
  pad_close = torch.exp(
    -(torch.square(err_fixed) + torch.square(err_moving)) / (0.018**2)
  )

  # A clamped 2.8-4.5 cm object stalls q_grip in [0.08, 0.38] rad while act_grip < 0.
  jaw_on_cube = torch.sigmoid((0.38 - q_grip) / 0.04) * torch.sigmoid(
    (q_grip - 0.08) / 0.02
  )
  preload = torch.sigmoid(-act_grip / 0.25)
  squeeze = jaw_on_cube * preload
  clamped = in_jaws * pad_close * (0.3 + 0.7 * squeeze)

  # Reward vertical lift off the table and bringing the clamped cube to the 3D goal.
  lift_z = ((obj_pos_w[:, 2] - 0.016) / 0.12).clamp(0.0, 1.0)
  pos_err_sq = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
  bringing = torch.exp(-pos_err_sq / (bringing_std**2))

  reward = (
    0.35 * open_approach
    + 0.5 * in_jaws
    + clamped * (1.0 + 3.0 * lift_z + 3.0 * bringing)
  )
  return reward * (0.5 + 0.5 * level_wrist)


def get_cube_spec(
  cube_size: float = 0.0175,
  mass: float = 0.025,
  rgba: tuple[float, float, float, float] = (0.85, 0.2, 0.2, 1.0),
) -> mujoco.MjSpec:
  """Create a 3.5 cm nominal graspable block (randomized 2.8-4.55 cm in DR)."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=rgba,
  )
  return spec


def so101_lift_cube_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the state-based SO-101 cube-lifting environment configuration."""
  cfg = make_lift_cube_env_cfg()
  cfg.sim.mujoco.iterations = 25
  cfg.sim.mujoco.ls_iterations = 50

  cfg.scene.entities = {
    "robot": get_so101_robot_cfg(),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = SO101_ACTION_SCALE

  # Adapt cube spawn and 3D goal workspace to the SO-101 ~30 cm reach.
  assert cfg.commands is not None
  lift_cmd = cfg.commands["lift_height"]
  assert isinstance(lift_cmd, LiftingCommandCfg)
  lift_cmd.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    x=(0.18, 0.26),
    y=(-0.10, 0.10),
    z=(0.015, 0.03),
    yaw=(-math.pi, math.pi),
  )
  lift_cmd.target_position_range = LiftingCommandCfg.TargetPositionRangeCfg(
    x=(0.18, 0.26),
    y=(-0.10, 0.10),
    z=(0.12, 0.20),
  )

  cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    "pinch_site",
  )
  cfg.rewards["lift"].params["asset_cfg"].site_names = ("pinch_site",)
  cfg.rewards["lift"].params["reaching_std"] = 0.08
  cfg.rewards["lift"].params["bringing_std"] = 0.18
  cfg.rewards["lift_precise"].params["std"] = 0.04
  cfg.rewards["grasp_and_lift"] = RewardTermCfg(
    func=so101_grasp_and_lift_reward,
    weight=2.0,
    params={
      "command_name": "lift_height",
      "object_name": "cube",
      "bringing_std": 0.18,
      "pad_cfg": SceneEntityCfg(
        "robot", geom_names=("fixed_jaw_pad", "moving_jaw_pad")
      ),
      "gripper_cfg": SceneEntityCfg("robot", joint_names=("gripper",)),
    },
  )
  cfg.rewards["joint_pos_limits"].params["asset_cfg"].joint_names = (
    "shoulder_.*",
    "elbow_flex",
    "wrist_.*",
  )
  cfg.rewards["joint_vel_hinge"].params["max_vel"] = 1.5
  cfg.rewards["action_rate_l2"].weight = -0.05

  fingertip_geoms = r"(fixed|moving)_jaw_pad"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  # Sim-to-real domain randomization across table/base heights, object sizes/masses, and servos.
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.015, 0.015),
    "y": (-0.015, 0.015),
    "z": (0.0, 0.055),
    "yaw": (-0.06, 0.06),
  }
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.08, 0.08)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.05, 0.05)
  cfg.events["cube_size"] = EventTermCfg(
    mode="startup",
    func=dr.geom_size,
    params={
      "asset_cfg": SceneEntityCfg("cube", geom_names=("cube_geom",)),
      "operation": "scale",
      "distribution": "uniform",
      "axes": [0, 1, 2],
      "ranges": (0.80, 1.30),
    },
  )
  cfg.events["cube_mass"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("cube", body_names=("cube",)),
      "alpha_range": (-0.35, 0.35),
      "distribution": "uniform",
    },
  )
  cfg.events["actuator_gains"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
      "kp_range": (0.85, 1.15),
      "kd_range": (0.85, 1.15),
      "operation": "scale",
      "distribution": "uniform",
    },
  )
  cfg.events["joint_frictionloss"] = EventTermCfg(
    mode="startup",
    func=dr.dof_frictionloss,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "operation": "scale",
      "distribution": "uniform",
      "ranges": (0.70, 1.30),
    },
  )
  cfg.events["effort_limits"] = EventTermCfg(
    mode="startup",
    func=dr.effort_limits,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
      "operation": "scale",
      "distribution": "uniform",
      "effort_limit_range": (0.85, 1.15),
    },
  )
  cfg.events["encoder_bias"] = EventTermCfg(
    mode="reset",
    func=dr.encoder_bias,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "bias_range": (-0.015, 0.015),
    },
  )

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "gripper"

  cfg.viewer.body_name = "base"
  cfg.viewer.distance = 0.75
  cfg.viewer.azimuth = 135.0
  cfg.viewer.elevation = -25.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.events["reset_base"].params["pose_range"] = {}
    cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
    cfg.events.pop("encoder_bias", None)
    lift_cmd.resampling_time_range = (4.0, 4.0)

  return cfg


def so101_lift_cube_vision_env_cfg(
  cam_type: Literal["rgb", "depth"],
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the wrist-camera (RGB or Depth) SO-101 cube-lifting configuration."""
  cfg = so101_lift_cube_env_cfg(play=play)

  camera_names = ["robot/wrist_cam"]
  cam_kwargs = {
    "robot/wrist_cam": {
      "height": 32,
      "width": 32,
    },
  }
  shared_cam_kwargs = dict(
    data_types=(cam_type,),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )

  cam_terms = {}
  for cam_name in camera_names:
    cam_cfg = CameraSensorCfg(
      name=cam_name.split("/")[-1],
      camera_name=cam_name,
      **cam_kwargs[cam_name],  # type: ignore[invalid-argument-type]
      **shared_cam_kwargs,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)
    param_kwargs: dict[str, Any] = {"sensor_name": cam_cfg.name}
    if cam_type == "depth":
      param_kwargs["cutoff_distance"] = 0.4
      func = manipulation_mdp.camera_depth
    else:
      func = manipulation_mdp.camera_rgb
    cam_terms[f"{cam_name.split('/')[-1]}_{cam_type}"] = ObservationTermCfg(
      func=func, params=param_kwargs
    )

  camera_obs = ObservationGroupCfg(
    terms=cam_terms, enable_corruption=False, concatenate_terms=True
  )
  cfg.observations["camera"] = camera_obs

  if cam_type == "rgb":
    cfg.events["cube_color"] = EventTermCfg(
      func=dr.geom_rgba,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("cube", geom_names=(".*",)),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0, 1, 2],
        "ranges": (0.0, 1.0),
      },
    )

  actor_obs = cfg.observations["actor"]
  actor_obs.terms.pop("ee_to_cube")
  actor_obs.terms.pop("cube_to_goal")

  actor_obs.terms["goal_position"] = ObservationTermCfg(
    func=manipulation_mdp.target_position,
    params={
      "command_name": "lift_height",
      "asset_cfg": SceneEntityCfg("robot", site_names=("pinch_site",)),
    },
  )

  return cfg
