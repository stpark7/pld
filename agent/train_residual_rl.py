"""
Algorithm-agnostic residual-RL orchestrator (PLD Stage 1, 5-phase loop).

Refactored from dice-rl ``agent/finetune/train_pld_stage1.py``. The orchestrator
drives the five phases by calling ONLY the ``ResidualRLAlgorithm`` contract
(``algo.base``) — it never references Cal-QL / SAC specifics. Swapping the
algorithm therefore needs only a new ``algo/<x>.py`` + a new cfg ``_target_``.

Phases:
  1. Offline collection : roll out the base policy (a_delta=0) until we have
     ``offline_success_episodes`` successful trajectories.
  2. Critic pretrain     : gated on ``algorithm.needs_pretrain``.
  3. Warm-up             : N episodes of base-only collection into the buffer.
  4. Online RL           : ``total_train_steps`` env steps, UTD updates, periodic
     eval + best-checkpoint saving.
  5. Final eval          : ``final_eval_episodes`` deterministic-residual episodes.

GPU/env safety: ONE shared 4-env pool (``util.env_pool.EnvPool``), reused for
both training and evaluation. ``evaluate()`` snapshots the collector chunk
state, resets the pool, runs eval on the same procs, then training resumes via a
fresh bootstrap with the chunk state restored (known-risk fix #2). An assertion
verifies ``next_a_base`` matches the base action actually executed at s'.
"""

from __future__ import annotations

import logging
import os
import time

import hydra
import numpy as np
import torch

from agent.train_agent import TrainAgent
from data.pld_data_collector import PLDDataCollector
from data.pld_action_normalizer import PLDActionNormalizer
from data.pld_replay_buffer import PLDReplayBuffer
from model.gr00t.gr00t_action_provider import GR00TActionProvider

log = logging.getLogger(__name__)


class TrainResidualRL(TrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        pld_cfg = cfg.pld

        self.xi = float(pld_cfg.xi)
        self.offline_success_target = int(pld_cfg.offline_success_episodes)
        self.critic_pretrain_steps = int(pld_cfg.critic_pretrain_steps)
        self.warmup_episodes = int(pld_cfg.warmup_episodes)
        self.total_train_steps = int(pld_cfg.total_train_steps)
        self.eval_every = int(pld_cfg.eval_every)
        self.eval_episodes_during = int(pld_cfg.eval_episodes_during)
        self.final_eval_episodes = int(pld_cfg.final_eval_episodes)
        self.utd = int(pld_cfg.utd)

        # GR00T provider (frozen base policy, shared)
        log.info(f"Loading GR00T checkpoint from {cfg.gr00t.model_path}")
        self.gr00t = GR00TActionProvider(
            model_path=cfg.gr00t.model_path,
            data_config=cfg.gr00t.get("data_config", "panda_omron"),
            embodiment_tag=cfg.gr00t.get("embodiment_tag", "new_embodiment"),
            denoising_steps=int(cfg.gr00t.get("denoising_steps", 4)),
            device=self.device,
            gr00t_repo_path=cfg.gr00t.get("gr00t_repo_path", None),
        )

        # Action normalizer
        self.normalizer = PLDActionNormalizer(
            cfg.normalization.lerobot_meta_dir,
            use_quantile=cfg.normalization.get("use_quantile", True),
        )

        self.action_chunk_size = int(cfg.gr00t.get("action_chunk_size", 16))

        # SINGLE collector reused for train + eval (chunk state is saved/restored
        # at every train<->eval boundary). task_description is owned by
        # RobocasaImageWrapper, so it is not piped through the collector.
        self.collector = PLDDataCollector(
            gr00t_provider=self.gr00t,
            normalizer=self.normalizer,
            n_envs=self.n_envs,
            action_chunk_size=self.action_chunk_size,
        )

        # Image / proprio shapes from shape_meta
        rgb_shape = tuple(cfg.shape_meta.obs.rgb.shape)  # (N_cam, H, W, 3)
        assert len(rgb_shape) == 4, f"rgb shape must be (N_cam, H, W, 3); got {rgb_shape}"
        self.n_cam = rgb_shape[0]
        self.image_hw = (rgb_shape[1], rgb_shape[2])
        self.proprio_dim = int(cfg.shape_meta.obs.state.shape[0])

        # Algorithm (the plug point). Instantiated from cfg.algorithm so the
        # orchestrator stays algorithm-agnostic.
        self.algorithm = hydra.utils.instantiate(cfg.algorithm)
        self.algorithm.build_optimizers(cfg)

        # Replay buffer (gamma comes from the algorithm contract).
        self.buffer = PLDReplayBuffer(
            max_size=int(cfg.train.buffer_capacity),
            n_cam=self.n_cam,
            image_hw=self.image_hw,
            proprio_dim=self.proprio_dim,
            action_dim=cfg.action_dim,
            sample_device=self.device,
            gamma=float(self.algorithm.gamma),
            storage_device=cfg.train.get("buffer_storage_device", "cpu"),
        )

        # Eval tracking
        self.best_eval_success = -1.0

    # ============================================================ phase 1: offline

    def collect_offline_trajectories(self, n_success_target: int):
        """Roll out the base policy (a_delta=0) until we have n_success_target
        successful trajectories.
        """
        log.info(
            f"Collecting offline trajectories — target {n_success_target} successes, "
            f"n_envs={self.n_envs}"
        )
        n_envs = self.n_envs
        venv = self.venv

        obs = venv.reset()
        self.collector.reset_chunks()

        traj_rgb = [[] for _ in range(n_envs)]
        traj_proprio = [[] for _ in range(n_envs)]
        traj_a_base = [[] for _ in range(n_envs)]
        traj_reward = [[] for _ in range(n_envs)]
        traj_success = [False] * n_envs

        successes: list = []
        total_episodes = 0

        # First step: zero action (raw) so we can read gr00t_raw from info.
        zero_raw = np.zeros((n_envs, 7), dtype=np.float32)
        next_obs, reward, terminated, truncated, info = venv.step(zero_raw)
        done = np.logical_or(np.asarray(terminated), np.asarray(truncated))
        gr00t_raw_list = PLDDataCollector.extract_gr00t_raw_from_info(info, n_envs)

        last_obs = next_obs
        last_gr00t_raw = gr00t_raw_list

        while len(successes) < n_success_target:
            a_base_norm = self.collector.get_a_base_norm(last_gr00t_raw, last_obs["rgb"])  # (n_envs, 7)

            a_total_norm = a_base_norm.copy()
            a_total_raw = self.collector.to_env_action(a_total_norm)

            cur_obs = last_obs
            cur_rgb = np.asarray(cur_obs["rgb"])
            cur_proprio = np.asarray(cur_obs["state"], dtype=np.float32)

            next_obs, reward, terminated, truncated, info = venv.step(a_total_raw)
            done = np.logical_or(np.asarray(terminated), np.asarray(truncated))
            next_gr00t_raw = PLDDataCollector.extract_gr00t_raw_from_info(info, n_envs)

            for j in range(n_envs):
                traj_rgb[j].append(cur_rgb[j])
                traj_proprio[j].append(cur_proprio[j])
                traj_a_base[j].append(a_base_norm[j])
                traj_reward[j].append(float(reward[j]))
                if reward[j] > 0:
                    traj_success[j] = True

            self.collector.advance(done)

            for j in range(n_envs):
                if done[j]:
                    total_episodes += 1
                    done_flags = [False] * len(traj_rgb[j])
                    if done_flags:
                        done_flags[-1] = True
                    if traj_success[j] and len(traj_rgb[j]) >= 2:
                        successes.append({
                            "rgb": np.stack(traj_rgb[j], axis=0),
                            "proprio": np.stack(traj_proprio[j], axis=0),
                            "a_base": np.stack(traj_a_base[j], axis=0),
                            "reward": np.asarray(traj_reward[j], dtype=np.float32),
                            "done": np.asarray(done_flags, dtype=bool),
                        })
                        log.info(
                            f"  offline progress: {len(successes)}/{n_success_target} successes "
                            f"({total_episodes} episodes attempted)"
                        )
                    traj_rgb[j] = []
                    traj_proprio[j] = []
                    traj_a_base[j] = []
                    traj_reward[j] = []
                    traj_success[j] = False

            last_obs = next_obs
            last_gr00t_raw = next_gr00t_raw

            if len(successes) >= n_success_target:
                break

        log.info(
            f"Offline collection done: {len(successes)} successes in "
            f"{total_episodes} episodes"
        )
        return successes

    # ============================================================ phase 2: critic pretrain

    def pretrain_critic(self, n_steps: int):
        """Critic-init via the algorithm contract (offline data only)."""
        log.info(f"Critic pretraining for {n_steps} steps (offline only)")
        for step in range(n_steps):
            batch = self.buffer.sample_offline_only(self.batch_size)
            info = self.algorithm.pretrain_critic_step(batch)

            if (step + 1) % 500 == 0 or step == 0:
                log.info(
                    f"  pretrain {step+1}/{n_steps}: "
                    f"critic_loss={float(info['critic_loss']):.4f} "
                    f"td={float(info['td_loss']):.4f} cql={float(info['cql_loss']):.4f} "
                    f"q_mean={float(info['q_data1_mean']):.3f} "
                    f"| mc_mean={float(info['mc_return_mean']):.4f} "
                    f"mc_max={float(info['mc_return_max']):.4f} "
                    f"mc_nz={float(info['mc_return_nonzero_frac']):.3f} "
                    f"floor_binds={float(info['calib_floor_binds_frac']):.3f} "
                    f"cql_diff=({float(info['cql_qf1_diff']):.3f},{float(info['cql_qf2_diff']):.3f})"
                )
                if self.use_wandb:
                    import wandb
                    wandb.log({f"pretrain/{k}": float(v) for k, v in info.items()},
                              step=step + 1)

    # ============================================================ collection helpers

    def _build_residual_action(self, last_obs, a_base_norm):
        """Compute a_delta (online, stochastic but non-reparameterized) for the
        residual policy via the algorithm contract."""
        rgb_t = torch.as_tensor(last_obs["rgb"], device=self.device)
        proprio_t = torch.as_tensor(
            last_obs["state"], dtype=torch.float32, device=self.device
        )
        a_base_t = torch.as_tensor(a_base_norm, device=self.device)
        a_delta_t = self.algorithm.select_action(
            rgb_t, proprio_t, a_base_t, deterministic=False
        )
        return a_delta_t.detach().cpu().numpy().astype(np.float32)

    def _collect_and_add(self, last_obs, last_gr00t_raw, use_residual: bool,
                         prev_next_a_base=None, prev_done=None):
        """Single env step (per env): build action, step env, add transition.

        Chunk-consistency assertion (known-risk fix #2): for envs that were NOT
        done on the previous step, the a_base computed here (the base action
        actually executed at this state s') must equal the ``next_a_base`` that
        was stored for the previous transition. A mismatch means the chunk
        pointer desynced.
        """
        n_envs = self.n_envs

        a_base_norm = self.collector.get_a_base_norm(last_gr00t_raw, last_obs["rgb"])  # (n_envs, 7)

        # ---- chunk-consistency assertion ----
        if prev_next_a_base is not None and prev_done is not None:
            for j in range(n_envs):
                if not prev_done[j]:
                    assert np.allclose(a_base_norm[j], prev_next_a_base[j], atol=1e-5), (
                        f"Chunk desync on env {j}: a_base at s' "
                        f"{a_base_norm[j]} != stored next_a_base "
                        f"{prev_next_a_base[j]}. The collector chunk pointer is "
                        "inconsistent with the executed base action."
                    )

        if use_residual:
            a_delta_norm = self._build_residual_action(last_obs, a_base_norm)
        else:
            a_delta_norm = np.zeros((n_envs, 7), dtype=np.float32)

        a_total_norm = a_base_norm + a_delta_norm
        a_total_raw = self.collector.to_env_action(a_total_norm)

        cur_rgb = np.asarray(last_obs["rgb"])
        cur_proprio = np.asarray(last_obs["state"], dtype=np.float32)

        next_obs, reward, terminated, truncated, info = self.venv.step(a_total_raw)
        done = np.logical_or(np.asarray(terminated), np.asarray(truncated))
        next_gr00t_raw = PLDDataCollector.extract_gr00t_raw_from_info(info, n_envs)

        # Advance chunk pointers, then read next_a_base (0 for done envs).
        self.collector.advance(done)
        next_a_base_norm = self.collector.get_next_a_base_norm(next_gr00t_raw, next_obs["rgb"], done)

        next_rgb = np.asarray(next_obs["rgb"])
        next_proprio = np.asarray(next_obs["state"], dtype=np.float32)
        reward_arr = np.asarray(reward, dtype=np.float32)
        done_arr = np.asarray(done, dtype=bool)

        self.buffer.add_online(
            rgb=cur_rgb,
            proprio=cur_proprio,
            a_base=a_base_norm,
            a_delta=a_delta_norm,
            a_total=a_total_norm,
            reward=reward_arr,
            next_rgb=next_rgb,
            next_proprio=next_proprio,
            next_a_base=next_a_base_norm,
            done=done_arr,
        )

        return next_obs, next_gr00t_raw, reward_arr, done_arr, next_a_base_norm

    def _bootstrap_first_obs(self):
        """Reset the shared pool, reset chunks, take a zero-action step to harvest
        gr00t_raw via info."""
        obs = self.venv.reset()
        self.collector.reset_chunks()
        zero_raw = np.zeros((self.n_envs, 7), dtype=np.float32)
        next_obs, _, terminated, truncated, info = self.venv.step(zero_raw)
        _ = np.logical_or(np.asarray(terminated), np.asarray(truncated))
        n = len(next_obs["state"])
        gr00t_raw = PLDDataCollector.extract_gr00t_raw_from_info(info, n)
        return next_obs, gr00t_raw

    # ============================================================ phase 3: warm-up

    def warmup_collect(self, n_episodes: int):
        log.info(f"Warm-up: collecting {n_episodes} episodes (a_delta=0)")
        last_obs, last_gr00t_raw = self._bootstrap_first_obs()
        completed = 0
        prev_next_a_base, prev_done = None, None
        while completed < n_episodes:
            (last_obs, last_gr00t_raw, reward, done,
             prev_next_a_base) = self._collect_and_add(
                last_obs, last_gr00t_raw, use_residual=False,
                prev_next_a_base=prev_next_a_base, prev_done=prev_done,
            )
            prev_done = done
            completed += int(done.sum())
        log.info(
            f"Warm-up done: {completed} episodes collected, "
            f"buffer size={self.buffer.online_size}"
        )

    # ============================================================ phase 4: online RL

    def train_loop(self):
        log.info(f"Main RL training: {self.total_train_steps} env steps (UTD={self.utd})")
        last_obs, last_gr00t_raw = self._bootstrap_first_obs()
        start_time = time.time()
        prev_next_a_base, prev_done = None, None

        for step in range(self.total_train_steps):
            # 1. Collect one step (per env) using residual policy.
            (last_obs, last_gr00t_raw, reward, done,
             prev_next_a_base) = self._collect_and_add(
                last_obs, last_gr00t_raw, use_residual=True,
                prev_next_a_base=prev_next_a_base, prev_done=prev_done,
            )
            prev_done = done

            # 2. Gradient updates via the algorithm contract (UTD + actor + temp + polyak).
            batch = self.buffer.sample(self.batch_size, expert_ratio=0.5)
            info = self.algorithm.update(batch, self.utd)

            if (step + 1) % 200 == 0:
                elapsed = time.time() - start_time
                log.info(
                    f"step {step+1}/{self.total_train_steps} "
                    f"| critic={float(info['critic_loss']):.3f} "
                    f"actor={float(info['actor_loss']):.3f} "
                    f"alpha={float(info['alpha']):.3f} buf={self.buffer.online_size} "
                    f"| q_data={float(info['q_data1_mean']):.3f} "
                    f"mc_nz={float(info['mc_return_nonzero_frac']):.3f} "
                    f"floor_binds={float(info['calib_floor_binds_frac']):.3f} "
                    f"cql_diff=({float(info['cql_qf1_diff']):.3f},{float(info['cql_qf2_diff']):.3f}) "
                    f"({(step+1)/elapsed:.2f} steps/s)"
                )
                if self.use_wandb:
                    import wandb
                    wandb.log({f"train/{k}": float(v) for k, v in info.items()},
                              step=step + 1)

            # Periodic eval + checkpoint. evaluate() snapshots+restores chunk
            # state and re-bootstraps the training loop on the shared pool.
            if (step + 1) % self.eval_every == 0:
                sr = self.evaluate(self.eval_episodes_during, deterministic=True)
                log.info(f"  eval @ step {step+1}: success_rate={sr:.4f}")
                if self.use_wandb:
                    import wandb
                    wandb.log({"eval/success_rate": sr}, step=step + 1)
                self._maybe_save_best(sr, step + 1)
                self.itr = step + 1
                self.save_model()
                # Resume training on the shared pool with a fresh bootstrap.
                last_obs, last_gr00t_raw = self._bootstrap_first_obs()
                prev_next_a_base, prev_done = None, None

    def _maybe_save_best(self, success_rate, step):
        if success_rate > self.best_eval_success:
            self.best_eval_success = success_rate
            path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(
                {
                    "step": step,
                    "success_rate": success_rate,
                    "model": self.algorithm.state_dict(),
                },
                path,
            )
            log.info(f"  saved BEST checkpoint (success={success_rate:.4f}) to {path}")

    # ============================================================ evaluation (shared pool)

    def evaluate(self, n_episodes: int, deterministic: bool = True) -> float:
        """Evaluate on the SAME shared 4-env pool used for training.

        Snapshots the collector chunk state, resets the pool, runs eval, then
        the caller resumes training via a fresh bootstrap. Restoring the chunk
        snapshot guarantees training's per-env pointers are not corrupted by the
        eval interlude (known-risk fix #2).
        """
        n_envs = self.n_envs
        venv = self.venv
        collector = self.collector

        # Snapshot chunk state so the training loop can be resumed cleanly.
        saved_chunk_state = collector.save_chunk_state()

        try:
            obs = venv.reset()
            collector.reset_chunks()
            zero_raw = np.zeros((n_envs, 7), dtype=np.float32)
            next_obs, _, terminated, truncated, info = venv.step(zero_raw)
            _ = np.logical_or(np.asarray(terminated), np.asarray(truncated))
            last_gr00t_raw = PLDDataCollector.extract_gr00t_raw_from_info(info, n_envs)
            last_obs = next_obs

            successes: list = []
            episode_success = [False] * n_envs

            max_eval_iter = max(
                2 * n_episodes * self.max_episode_steps // max(n_envs, 1), 500
            )
            iter_count = 0

            while len(successes) < n_episodes:
                if iter_count >= max_eval_iter:
                    log.warning(
                        f"Eval hit safety cap {max_eval_iter} iters with only "
                        f"{len(successes)}/{n_episodes} episodes completed. "
                        "Returning partial result."
                    )
                    break
                iter_count += 1
                a_base_norm = collector.get_a_base_norm(last_gr00t_raw, last_obs["rgb"])
                rgb_t = torch.as_tensor(last_obs["rgb"], device=self.device)
                proprio_t = torch.as_tensor(
                    last_obs["state"], dtype=torch.float32, device=self.device
                )
                a_base_t = torch.as_tensor(a_base_norm, device=self.device)
                a_delta_t = self.algorithm.select_action(
                    rgb_t, proprio_t, a_base_t, deterministic=deterministic
                )
                a_delta_norm = a_delta_t.detach().cpu().numpy().astype(np.float32)
                a_total_norm = a_base_norm + a_delta_norm
                a_total_raw = collector.to_env_action(a_total_norm)

                next_obs, reward, terminated, truncated, info = venv.step(a_total_raw)
                done = np.logical_or(np.asarray(terminated), np.asarray(truncated))
                next_gr00t_raw = PLDDataCollector.extract_gr00t_raw_from_info(info, n_envs)
                collector.advance(done)

                for j in range(n_envs):
                    if reward[j] > 0:
                        episode_success[j] = True
                    if done[j]:
                        successes.append(bool(episode_success[j]))
                        episode_success[j] = False
                        if len(successes) >= n_episodes:
                            break

                last_obs = next_obs
                last_gr00t_raw = next_gr00t_raw

            sr = float(np.mean(successes[:n_episodes])) if successes else 0.0
        finally:
            # Restore chunk state regardless of how eval exits.
            collector.restore_chunk_state(saved_chunk_state)

        return sr

    # ============================================================ orchestration

    def run(self):
        log.info("===== PLD residual RL: phase 1 (offline trajectory collection) =====")
        offline_trajs = self.collect_offline_trajectories(self.offline_success_target)
        self.buffer.load_offline_trajectories(offline_trajs)

        if self.algorithm.needs_pretrain:
            log.info("===== PLD residual RL: phase 2 (critic pretraining) =====")
            self.pretrain_critic(self.critic_pretrain_steps)
        else:
            log.info("===== PLD residual RL: phase 2 SKIPPED (algorithm.needs_pretrain=False) =====")

        log.info("===== PLD residual RL: phase 3 (warm-up) =====")
        self.warmup_collect(self.warmup_episodes)

        log.info("===== PLD residual RL: phase 4 (online RL) =====")
        self.train_loop()

        log.info("===== PLD residual RL: phase 5 (final evaluation) =====")
        final_sr = self.evaluate(self.final_eval_episodes, deterministic=True)
        log.info(
            f"FINAL success rate ({self.final_eval_episodes} episodes): {final_sr:.4f}"
        )
        if self.use_wandb:
            import wandb
            wandb.log({"eval/final_success_rate": final_sr})
        return final_sr
