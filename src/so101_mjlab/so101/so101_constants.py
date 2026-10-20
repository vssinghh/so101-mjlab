"""SO-ARM101 constants and robot configuration for mjlab."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

_HERE = Path(__file__).parent

SO101_XML: Path = _HERE / "xmls" / "so101.xml"
assert SO101_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(SO101_XML))


##
# Actuator config (Feetech STS3215 12V 30 kg.cm servo, system-identified at 50 Hz).
# Fitted against 50 Hz hardware step responses and gravity droop probes
# (checkpoints/sysid_sts3215.json, mean trajectory RMSE = 0.0097 rad = 0.55 deg).
##

ARM_EFFORT_LIMIT = 1.47  # 50% hardware Torque_Limit register (500/1000 of 2.94 Nm)
GRIPPER_EFFORT_LIMIT = 1.20  # 40% hardware Torque_Limit register (400/1000 of 2.94 Nm)
ARMATURE = 0.0211
FRICTIONLOSS = 0.20
STIFFNESS = 18.33
DAMPING = 1.34

SO101_ARM_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
  ),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=ARM_EFFORT_LIMIT,
  armature=ARMATURE,
  frictionloss=FRICTIONLOSS,
  delay_min_lag=0,
  delay_max_lag=2,
  delay_hold_prob=0.2,
)

SO101_GRIPPER_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("gripper",),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=GRIPPER_EFFORT_LIMIT,
  armature=ARMATURE,
  frictionloss=FRICTIONLOSS,
  delay_min_lag=0,
  delay_max_lag=2,
  delay_hold_prob=0.2,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.005),
  joint_pos={
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.35,
    "elbow_flex": 0.70,
    "wrist_flex": 0.95,
    "wrist_roll": -1.5708,
    "gripper": 0.65,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision", ".*_pad", "gripper_palm_col"),
  contype=1,
  conaffinity=1,
  condim={
    ".*_pad": 6,
    ".*": 3,
  },
  friction={
    ".*_pad": (1.5, 5e-3, 5e-4),
    ".*": (0.8,),
  },
  solref={
    ".*_pad": (0.008, 1.0),
  },
  priority={
    ".*_pad": 1,
    ".*": 0,
  },
)

##
# Final config.
##

SO101_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(SO101_ARM_ACTUATOR_CFG, SO101_GRIPPER_ACTUATOR_CFG),
  soft_joint_pos_limit_factor=0.9,
)


def get_so101_robot_cfg() -> EntityCfg:
  """Return a fresh SO-101 EntityCfg instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=SO101_ARTICULATION,
  )


SO101_ACTION_SCALE: dict[str, float] = {
  "shoulder_pan": 0.5,
  "shoulder_lift": 0.5,
  "elbow_flex": 0.5,
  "wrist_flex": 0.5,
  "wrist_roll": 0.5,
  "gripper": 0.5,
}


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_so101_robot_cfg())
  viewer.launch(robot.spec.compile())
