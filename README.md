# so101-mjlab

Manipulation environments, GPU reinforcement learning training, and 50 Hz sim-to-real deployment for the open-source 6-DoF **SO-ARM101** robot arm built on [`mjlab`](https://github.com/mujocolab/mjlab).

<p align="center">
  <img src="docs/so101_trained_lift_rollout.gif" width="48%" alt="Trained SO-101 lifting rollout in MuJoCo" />
  <img src="docs/so101_real_lift_side.gif" width="48%" alt="Physical SO-101 sim-to-real lift rollout" />
</p>
<p align="center">
  <img src="docs/so101_sim2real_lift.gif" width="97%" alt="Physical SO-101 dual-camera (side + wrist) lift rollout" />
</p>

## Registered Tasks

Installing `so101-mjlab` registers three manipulation tasks via the `mjlab.tasks` entry point:

| Task ID | Observation Space | Status & Description |
| :--- | :--- | :--- |
| `Mjlab-Lift-Cube-SO101` | 24-D state | **Trained & Sim-to-Real Verified** (`checkpoints/policy.onnx`). Reach, grasp, and lift 2.8 to 4.55 cm objects (cubes and compliant sponges) to randomized 3D targets using the SO-101 rotary jaw gripper. |
| `Mjlab-Lift-Cube-SO101-Rgb` | State + 32x32 RGB (`wrist_cam`) | Environment config for visual lifting with domain-randomized object colors and Spatial Softmax CNN policy (no pre-trained checkpoint shipped). |
| `Mjlab-Lift-Cube-SO101-Depth` | State + 32x32 Depth (`wrist_cam`) | Environment config for depth-based visual lifting from the wrist-mounted camera (no pre-trained checkpoint shipped). |

## Quick Start

### 1. Preview the SO-101 Robot in MuJoCo

```bash
uv run python -m so101_mjlab.so101.so101_constants
```

### 2. Interactive Play & Evaluation

Evaluate with a zero policy in the `mjlab` viewer:

```bash
uv run play Mjlab-Lift-Cube-SO101 --agent zero
```

Verify the shipped ONNX policy (`checkpoints/policy.onnx`) in a standalone MuJoCo rollout:

```bash
uv run so101-deploy --dry-run --onnx checkpoints/policy.onnx
```

Or evaluate a PyTorch checkpoint produced by a training run in the interactive viewer:

```bash
uv run play Mjlab-Lift-Cube-SO101 --checkpoint-file checkpoints/lift_v7_sysid/latest_model.pt
```

### 3. Training (Local CUDA GPU or Automated Remote Google Colab GPU)

Train on a remote Google Colab GPU from the terminal (`2048` parallel environments for `1000` PPO iterations, or `49.15M` environment steps; uploads a repo snapshot, runs `mjlab` PPO training, exports `policy.onnx`, and downloads the artifacts into `checkpoints/<run_name>/`):

```bash
./scripts/train.sh lift_v7_sysid --colab --env.scene.num-envs 2048 --agent.max-iterations 1000
```

Or train directly on a local CUDA workstation:

```bash
uv run train Mjlab-Lift-Cube-SO101 --env.scene.num-envs 2048 --agent.max-iterations 1000
```

### 4. Physical SO-101 Hardware Deployment (macOS / Linux)

#### Servo Calibration & System Identification
[`sts3215_driver.py`](src/so101_mjlab/sts3215_driver.py) automatically loads zero-radian tick offsets from your LeRobot follower calibration file at `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so_101_follower.json` (or a custom path via `SO101_CALIBRATION_FILE`). You can also override the serial port via `--port` or `SO101_PORT`.

The MuJoCo actuator model in [`so101_constants.py`](src/so101_mjlab/so101/so101_constants.py) uses parameters fitted directly against 50 Hz hardware step responses and static gravity droop probes ([`checkpoints/sysid_sts3215.json`](checkpoints/sysid_sts3215.json), mean trajectory RMSE `0.0097 rad` or `0.55 deg`):
- `STIFFNESS = 18.33 N*m/rad`, `DAMPING = 1.34 N*m*s/rad`
- `FRICTIONLOSS = 0.20 N*m`, `ARMATURE = 0.0211 kg*m^2`
- `ARM_EFFORT_LIMIT = 1.47 N*m` (`50%` of STS3215 `2.94 N*m` stall torque), `GRIPPER_EFFORT_LIMIT = 1.20 N*m` (`40%`)

To re-run 50 Hz step-response system identification on your physical follower arm:

```bash
uv run python scripts/sysid_sts3215.py --port /dev/tty.usbmodem5B790143431
```

#### Running the Policy at 50 Hz
Deploy the exported [`checkpoints/policy.onnx`](checkpoints/policy.onnx) policy (`lift_v7_sysid`, trained for `1,000` PPO iterations or `49.15M` steps across `2,048` environments) at 50 Hz over 1 Mbps Feetech STS3215 USB serial:

```bash
# Dry-run physics and ONNX verification without hardware plugged in
uv run so101-deploy --dry-run --onnx checkpoints/policy.onnx

# Live deployment on physical SO-101 follower arm (lifting a 4.2 cm sponge or cube)
uv run so101-deploy --onnx checkpoints/policy.onnx --port /dev/tty.usbmodem5B790143431 --cube-xyz 0.215 0.002 -0.036
```

**Deployment notes (`deploy_real.py`)**:
- **Closed-loop state & actions**: At each 50 Hz tick, [`deploy_real.py`](src/so101_mjlab/deploy_real.py) reads the 6 STS3215 encoders (`q_cur`), computes EMA-filtered finite-difference joint velocities (`dq_est`), evaluates `policy.onnx` on the raw unclamped joint state, and sends the policy's 6 joint position targets (including the gripper) directly to the servos.
- **Object pose input**: Because `Mjlab-Lift-Cube-SO101` is a state-based policy (24-D observation without camera inputs), the initial 3D object position is provided via `--cube-xyz` in the robot base frame; once the jaws clamp and stall on the object at the target location, the object position observation tracks the midpoint between the jaw pads (`pad_mid`).
- **Torque pre-sync, safety clamping & reset**: Before enabling servo torque (`reg 40 = 1`), [`configure_servo_torque`](src/so101_mjlab/sts3215_driver.py) syncs `Goal_Position = Present_Position` (`reg 42 = reg 56`) and programs hardware acceleration and velocity profiles (`reg 41 = 35`, `reg 46 = 1100`) to prevent startup jerk. Commands are clipped to joint limits and a per-step rate limit (`--max-delta-rad 0.030` with a 10-step soft ramp). After the 105-step RL rollout completes, the wrapper interpolates back to release the object and return to `HOME_KEYFRAME` (disable via `--no-place-back`).

## License & Attribution

Licensed under the [Apache License, Version 2.0](LICENSE).

The SO-ARM101 STL meshes and base MJCF kinematic structure in [`src/so101_mjlab/so101/xmls/`](src/so101_mjlab/so101/xmls/) are derived from [`TheRobotStudio/SO-ARM100`](https://github.com/TheRobotStudio/SO-ARM100) (Apache 2.0).
