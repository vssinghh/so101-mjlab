"""SO-ARM101 manipulation tasks registered as an mjlab plugin package."""

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  so101_lift_cube_env_cfg,
  so101_lift_cube_vision_env_cfg,
)
from .rl_cfg import (
  so101_lift_cube_ppo_runner_cfg,
  so101_lift_cube_vision_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-SO101",
  env_cfg=so101_lift_cube_env_cfg(),
  play_env_cfg=so101_lift_cube_env_cfg(play=True),
  rl_cfg=so101_lift_cube_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-SO101-Rgb",
  env_cfg=so101_lift_cube_vision_env_cfg(cam_type="rgb"),
  play_env_cfg=so101_lift_cube_vision_env_cfg(cam_type="rgb", play=True),
  rl_cfg=so101_lift_cube_vision_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-SO101-Depth",
  env_cfg=so101_lift_cube_vision_env_cfg(cam_type="depth"),
  play_env_cfg=so101_lift_cube_vision_env_cfg(cam_type="depth", play=True),
  rl_cfg=so101_lift_cube_vision_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
