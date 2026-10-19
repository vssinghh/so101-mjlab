"""SO-ARM101 constants and robot configuration for mjlab."""

import math
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
# Actuator config (Feetech STS3215 12V 30 kg.cm servo).
##

EFFORT_LIMIT = 3.35
ARMATURE = 0.028

NATURAL_FREQ = 6.0 * 2.0 * math.pi  # 6 Hz closed-loop bandwidth
DAMPING_RATIO = 1.2

STIFFNESS = ARMATURE * NATURAL_FREQ**2
DAMPING = 2.0 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ

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
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE,
)

SO101_GRIPPER_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("gripper",),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE,
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
    "gripper": 0.38,
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
