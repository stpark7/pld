# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PLD Stage-1 residual RL: fine-tune a **frozen GR00T N1.5** base policy on a RoboCasa
robot-manipulation task by learning an additive **residual** action with **Cal-QL + SAC**.
Code was refactored out of a larger `dice-rl` repo (many docstrings still reference it).

The base policy is never trained. A small residual actor `a_delta ∈ [-xi, xi]^7` is learned
on top of GR00T's base action `a_base`; the executed action is `a_total = a_base + a_delta`.

## Running

Requires the `robocasa` conda env (heavy deps: robocasa, gymnasium, mujoco/EGL, GR00T, torch,
torchvision, hydra, omegaconf, einops, wandb). There is **no** requirements/setup file; deps are
assumed already installed in that env. GR00T is imported lazily and can be made importable via
`gr00t.gr00t_repo_path` in the config without pip-installing it.

Required env vars (referenced by configs via `${oc.env:...}`):
- `DICE_RL_LOG_DIR` — root for run output / checkpoints (always needed).
- `DICE_RL_WANDB_ENTITY` — only when wandb is enabled (set `wandb: null` to disable).

Training (Hydra entry point; `config_path` is `<cwd>/cfg`, so run from repo root):
```
python script/run.py --config-name robocasa/finetune/CloseToasterOvenDoor/calql_sac
# fast plumbing check (tiny phase sizes):
python script/run.py --config-name robocasa/finetune/CloseToasterOvenDoor/calql_sac_smoke
```
The `calql_sac_smoke.yaml` config is the closest thing to a test — it runs the full 5-phase loop
end-to-end with minimal sizes. There is no unit-test suite, linter, or CI configured.

Standalone evaluation of a trained checkpoint (does NOT use Hydra — it manually expands the
config `defaults:` list):
```
python script/eval_residual_policy.py \
  --config cfg/robocasa/finetune/CloseToasterOvenDoor/calql_sac.yaml \
  --checkpoint $DICE_RL_LOG_DIR/.../checkpoint/best.pt \
  --n_episodes 100 --n_envs 4
```

## Architecture

### The plug point (read this first)
`agent/train_residual_rl.py::TrainResidualRL` is an **algorithm-agnostic** orchestrator. It only
ever calls the `algo/base.py::ResidualRLAlgorithm` abstract contract:
`build_optimizers`, `select_action`, `pretrain_critic_step`, `update`, `state_dict`/`load_state_dict`,
and the properties `gamma` / `needs_pretrain` / `needs_mc_return`.

To swap the RL algorithm you add a new `algo/<x>.py` implementing that contract plus a new cfg
with a different `algorithm._target_`. **Do not** put algorithm-specific logic (losses, log_alpha,
optimizer wiring) in the orchestrator — it lives entirely in the algorithm subclass. The current
implementation is `algo/calql_sac.py::PLDCalQLSAC`, which owns the encoder/actor/critic as
submodules so one `state_dict()` round-trips the whole graph + the learnable `log_alpha`.

### Five-phase training loop (`TrainResidualRL.run`)
1. **Offline collection** — roll out base policy (`a_delta=0`) until `offline_success_episodes`
   successful trajectories are gathered. Per-step MC returns are computed when loaded into the buffer.
2. **Critic pretrain** — gated on `algorithm.needs_pretrain`; offline-only Cal-QL critic init.
3. **Warm-up** — `warmup_episodes` of base-only collection into the buffer.
4. **Online RL** — `total_train_steps` env steps; each step does UTD critic updates + actor + temp
   + polyak; periodic eval saves `best.pt` on improvement and `state_<step>.pt` checkpoints.
5. **Final eval** — `final_eval_episodes` with the deterministic residual.

### The 4-env shared pool — the central safety invariant
`util/env_pool.py` enforces `MAX_ROBOCASA_ENVS = 4` via a module-level assert. RoboCasa is heavy;
>4 concurrent worker processes **freeze the machine**. The original dice-rl code built a *separate*
eval venv (doubling processes) — that footgun is deliberately removed. There is exactly **ONE**
vectorized env, reused for both training and eval.

Because train and eval share one env + one collector, `evaluate()` snapshots the collector's
per-env chunk state (`save_chunk_state`), resets the pool, runs eval, restores the snapshot, and the
caller resumes training via a fresh `_bootstrap_first_obs()`. Corrupting this would desync the
`(s, a_base, s', next_a_base)` tuples and the TD target.

### Action / observation plumbing (subtle, easy to break)
GR00T emits a **16-step action chunk** but the env steps one action at a time. `PLDDataCollector`
caches the chunk per-env and refills when exhausted or on episode reset (`advance(done)` resets the
pointer for done envs). Action flow per step:
- GR00T → 12-D RoboCasa dict action → extract **7-D arm** (`ee_pos[3], ee_rot[3], gripper[1]`)
  → `PLDActionNormalizer.normalize` (q01/q99 from a lerobot `stats.json`) → `a_base ∈ [-1,1]^7`.
- residual `a_delta`, then `a_total = a_base + a_delta` → `unnormalize` → env `step`.
- `RobocasaImageWrapper._action_to_dict` converts the 7-D arm action back to the 12-D dict
  (base_motion and control_mode zeroed).

Observations: the 9-D proprio `state` and 3-camera `rgb` `(N_cam,H,W,3)` go to the residual policy's
encoder/critic. `RobocasaImageWrapper` also puts the **non-image** part of a GR00T-N1.5 obs (5 state
keys incl. base pose + language + a `_video_keys` camera-order hint) in **`info["gr00t_raw"]`** every
step. The camera frames are **not** duplicated there — they are the same pixels as `obs["rgb"]`, so
`PLDDataCollector._assemble_gr00t_obs` rebuilds GR00T's per-camera video inputs from `obs["rgb"]`
(hence `get_a_base_norm`/`get_next_a_base_norm` take an `rgb` arg). This relies on
`keep_cams_separate=True` and `camera_keys == GR00T_VIDEO_KEYS` (asserted in the wrapper).
`extract_gr00t_raw_from_info` **raises** if the key is missing (silent fallback hides bugs).

There is a chunk-consistency **assertion** in `_collect_and_add`: for non-done envs, the `a_base`
recomputed at `s'` must equal the `next_a_base` stored for the previous transition. A failure means
the chunk pointer desynced.

### Replay buffer (`data/pld_replay_buffer.py`)
RLPD-style 50/50 online+offline sampling (`expert_ratio=0.5`). Online is a ring buffer; offline is
loaded once and never evicted. Images stored as **uint8 on CPU** to bound VRAM, moved to
`sample_device` per batch. Offline transitions carry MC returns (the Cal-QL calibration floor);
online transitions get `mc_return = 0` (a non-binding floor under sparse 0/1 rewards).

### Models
- `model/encoder/resnet18_encoder.py` — one ImageNet ResNet18 per camera (typically frozen
  backbone + frozen BN), features concatenated and projected to 256-D. Random-shift aug in train mode.
- `model/actor/residual_gaussian_actor.py` — tanh-squashed Gaussian, output scaled to `[-xi,xi]`,
  conditioned on `(image_feat, proprio, a_base)`; SAC-style log-prob with tanh + xi-scale correction.
- `model/critic/critic.py::CriticObsAct` — double-Q state-action critic. `cond_dim = feat(256) +
  proprio(9) = 265`; the critic is "state-only" in interface (image feat is baked into the cond).

### Config layout (Hydra)
`cfg/robocasa/finetune/<Task>/`: `_base_residual_rl.yaml` holds everything algorithm-agnostic
(env, gr00t paths, normalization, schedule, buffer). Variant files (`calql_sac.yaml`,
`calql_sac_smoke.yaml`) inherit it via `defaults:` and add only the `algorithm:` subtree. Custom
OmegaConf resolvers `${eval:...}`, `${round_up:...}`, `${round_down:...}` are registered in
`script/run.py` (and re-registered, plus `${now:}`, in the standalone eval script).

`env/gym_utils/make_async` only supports `env_type == "robocasa"`; the other dice-rl sim backends
(pusht, furniture, robomimic, d3il) and their wrappers were intentionally not ported and will raise.