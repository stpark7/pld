"""
PLD data collector.

Owns a single frozen GR00TActionProvider and a per-env action-chunk cache:
GR00T returns 16-step chunks but the environment steps one action at a time.
We refill a per-env cache when it's exhausted (or when the env resets).

Action plumbing:
  GR00T -> dict action (raw, 12-D RoboCasa space, but we only use 7-D arm)
       -> extract arm-only (ee_pos, ee_rot, gripper) -> raw 7-D
       -> normalize via PLDActionNormalizer -> a_base in [-1, 1]^7
  Residual policy -> a_delta in [-xi, xi]^7  (normalized space)
  a_total = a_base + a_delta  (normalized; may exceed [-1, 1])
       -> unnormalize -> raw 7-D
       -> wrapper.step (which converts to 12-D dict, base_motion=0)

Observation plumbing:
  RobocasaImageWrapper exposes the NON-image part of a GR00T N1.5 obs (5 state
  keys + language + a ``_video_keys`` camera-order hint) via ``info["gr00t_raw"]``.
  The camera frames are NOT duplicated there — they are the same pixels as
  ``obs["rgb"]`` — so this collector reconstructs GR00T's per-camera video inputs
  from ``obs["rgb"]`` (``_assemble_gr00t_obs``) before querying GR00T.

Chunk-state save/restore:
  Because train and eval share ONE collector + 4-env pool, the orchestrator
  must be able to snapshot and reset the per-env chunk cache at every
  train<->eval boundary (see save_chunk_state / restore_chunk_state). Pointer
  desync would corrupt the (s, a_base, s', next_a_base) tuples and TD target.
"""

import copy
import logging

import numpy as np

from model.gr00t.gr00t_action_provider import GR00TActionProvider
from data.pld_action_normalizer import PLDActionNormalizer

log = logging.getLogger(__name__)


class PLDDataCollector:
    def __init__(
        self,
        gr00t_provider: GR00TActionProvider,
        normalizer: PLDActionNormalizer,
        n_envs: int,
        action_chunk_size: int = 16,
    ):
        self.gr00t = gr00t_provider
        self.normalizer = normalizer
        self.n_envs = n_envs
        self.chunk_size = action_chunk_size

        # Per-env chunk cache (raw arm 7-D actions, shape (chunk_size, 7))
        self.chunks: list = [None] * n_envs
        # chunk_idx[j] = chunk_size means "needs refill"
        self.chunk_idx: list = [self.chunk_size] * n_envs

    # ---- chunk management ----

    def reset_chunks(self, env_indices=None):
        if env_indices is None:
            env_indices = range(self.n_envs)
        for j in env_indices:
            self.chunks[j] = None
            self.chunk_idx[j] = self.chunk_size

    def save_chunk_state(self) -> dict:
        """Snapshot the mutable per-env chunk cache so the orchestrator can
        restore it after a train<->eval boundary (known-risk fix #2).

        Returns a deep copy so subsequent advance()/refill() can't mutate the
        snapshot in place.
        """
        return {
            "chunks": copy.deepcopy(self.chunks),
            "chunk_idx": list(self.chunk_idx),
            "chunk_size": self.chunk_size,
        }

    def restore_chunk_state(self, state: dict):
        """Restore a snapshot taken by save_chunk_state()."""
        self.chunks = copy.deepcopy(state["chunks"])
        self.chunk_idx = list(state["chunk_idx"])
        self.chunk_size = state["chunk_size"]

    def _assemble_gr00t_obs(self, partial_obs: dict, rgb_env: np.ndarray) -> dict:
        """Reconstruct the full GR00T observation for one env.

        ``partial_obs`` is the video-less dict produced by RobocasaImageWrapper
        (5 state keys + language + ``_video_keys``). ``rgb_env`` is that env's
        ``obs["rgb"]`` of shape (N_cam, H, W, 3). The per-camera GR00T video
        inputs are rebuilt here from the same pixels as ``obs["rgb"]`` (with the
        ``T=1`` axis GR00T expects) so the image is never shipped twice.
        """
        video_keys = partial_obs["_video_keys"]
        rgb_env = np.asarray(rgb_env)
        if rgb_env.shape[0] != len(video_keys):
            raise ValueError(
                f"obs rgb has {rgb_env.shape[0]} cameras but gr00t_raw advertises "
                f"{len(video_keys)} video keys {video_keys}"
            )
        out = {k: v for k, v in partial_obs.items() if k != "_video_keys"}
        for i, vk in enumerate(video_keys):
            out[vk] = rgb_env[i][None]  # (H, W, 3) -> (T=1, H, W, 3)
        return out

    def _refill_chunk(self, env_idx: int, gr00t_obs: dict):
        chunk_dict = self.gr00t.get_action(gr00t_obs)
        arm_raw = self.normalizer.extract_arm_from_gr00t(chunk_dict)  # (T, 7)
        if arm_raw.shape[0] != self.chunk_size:
            # GR00T chunk size mismatch — log once and adapt
            log.warning(
                f"GR00T returned chunk of size {arm_raw.shape[0]}, expected {self.chunk_size}. "
                "Using returned size."
            )
            self.chunk_size = arm_raw.shape[0]
        self.chunks[env_idx] = arm_raw
        self.chunk_idx[env_idx] = 0

    # ---- main API ----

    def get_a_base_norm(self, gr00t_obs_per_env: list, rgb: np.ndarray) -> np.ndarray:
        """
        Return a_base (normalized, 7-D) for the current obs of each env. Refills
        GR00T chunks for envs whose cache is exhausted.

        Args:
            gr00t_obs_per_env: list of length n_envs with the video-less GR00T obs
                dicts (5 state keys + language + ``_video_keys``) produced by
                RobocasaImageWrapper into ``info["gr00t_raw"]``.
            rgb: (n_envs, N_cam, H, W, 3) ``obs["rgb"]`` for the current step. Used
                to rebuild GR00T's per-camera video inputs without a second image copy.
        """
        rgb = np.asarray(rgb)
        for j in range(self.n_envs):
            if self.chunk_idx[j] >= self.chunk_size:
                full_obs = self._assemble_gr00t_obs(gr00t_obs_per_env[j], rgb[j])
                self._refill_chunk(j, full_obs)

        out = np.zeros((self.n_envs, 7), dtype=np.float32)
        for j in range(self.n_envs):
            raw = self.chunks[j][self.chunk_idx[j]]
            out[j] = self.normalizer.normalize(raw)
        return out

    def advance(self, done: np.ndarray):
        """Advance each env's chunk pointer after env.step."""
        for j in range(self.n_envs):
            if done[j]:
                self.chunks[j] = None
                self.chunk_idx[j] = self.chunk_size
            else:
                self.chunk_idx[j] += 1

    def get_next_a_base_norm(
        self,
        gr00t_obs_per_env_next: list,
        next_rgb: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        """
        Return a_base for the NEXT state (used as `next_a_base` in the buffer).
        Must be called AFTER `advance(done)`. For done envs, returns 0 (will be
        masked by (1 - done) in the Bellman target).

        ``next_rgb`` is the (n_envs, N_cam, H, W, 3) next-step ``obs["rgb"]``, used
        to rebuild GR00T's per-camera video inputs on a chunk refill.
        """
        next_rgb = np.asarray(next_rgb)
        out = np.zeros((self.n_envs, 7), dtype=np.float32)
        for j in range(self.n_envs):
            if done[j]:
                continue
            if self.chunk_idx[j] >= self.chunk_size:
                full_obs = self._assemble_gr00t_obs(
                    gr00t_obs_per_env_next[j], next_rgb[j]
                )
                self._refill_chunk(j, full_obs)
            raw = self.chunks[j][self.chunk_idx[j]]
            out[j] = self.normalizer.normalize(raw)
        return out

    # ---- env action helpers ----

    def to_env_action(self, a_total_norm: np.ndarray) -> np.ndarray:
        """Convert normalized total action (n_envs, 7) to raw 7-D for env.step()."""
        return np.stack(
            [self.normalizer.unnormalize(a) for a in a_total_norm], axis=0
        ).astype(np.float32)

    @staticmethod
    def extract_gr00t_raw_from_info(info, n_envs: int) -> list:
        """
        Pull the GR00T-ready obs dict (key ``gr00t_raw``) out of an env-step info
        structure produced by RobocasaImageWrapper. Supports:
          - list of per-env dicts (SyncVectorEnv);
          - dict with key "gr00t_raw" -> list of per-env dicts (gymnasium-style);
          - tuple variants.
        Raises if the key is missing on any env — silent fallback masks bugs.
        """
        if isinstance(info, (list, tuple)):
            out = []
            for j in range(n_envs):
                if not isinstance(info[j], dict) or "gr00t_raw" not in info[j]:
                    raise KeyError(
                        f"info[{j}] missing 'gr00t_raw'. info[{j}].keys()="
                        f"{list(info[j].keys()) if isinstance(info[j], dict) else type(info[j])}"
                    )
                out.append(info[j]["gr00t_raw"])
            return out
        if isinstance(info, dict):
            if "gr00t_raw" not in info:
                raise KeyError(
                    f"info missing 'gr00t_raw'. info.keys()={list(info.keys())}"
                )
            raw = info["gr00t_raw"]
            if isinstance(raw, (list, tuple)):
                return list(raw)
            return [raw[j] for j in range(n_envs)]
        raise TypeError(f"Unrecognised info type: {type(info)}")
