"""
Per-step replay buffer for PLD Stage 1 residual RL.

Stores image + proprio + (a_base, a_delta, a_total) + reward + next image/proprio
+ next_a_base + done. Supports RLPD-style 50/50 online+offline sampling.

Images are stored on CPU as uint8 to minimise GPU VRAM. Sampled minibatches are
moved to the requested device.
"""

import logging
import numpy as np
import torch

log = logging.getLogger(__name__)


class PLDReplayBuffer:
    def __init__(
        self,
        max_size: int,
        n_cam: int,
        image_hw: tuple,
        proprio_dim: int,
        action_dim: int,
        sample_device: str = "cuda",
        gamma: float = 0.99,
        storage_device: str = "cpu",
    ):
        self.max_size = max_size
        self.n_cam = n_cam
        self.image_hw = image_hw
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.sample_device = sample_device
        self.storage_device = storage_device
        self.gamma = gamma

        H, W = image_hw

        def img_buf():
            return torch.zeros(
                (max_size, n_cam, H, W, 3), dtype=torch.uint8, device=storage_device
            )

        def vec(dim, dtype=torch.float32):
            return torch.zeros((max_size, dim), dtype=dtype, device=storage_device)

        # Online ring buffer
        self.online_rgb = img_buf()
        self.online_proprio = vec(proprio_dim)
        self.online_a_base = vec(action_dim)
        self.online_a_delta = vec(action_dim)
        self.online_a_total = vec(action_dim)
        self.online_reward = torch.zeros((max_size,), dtype=torch.float32, device=storage_device)
        self.online_next_rgb = img_buf()
        self.online_next_proprio = vec(proprio_dim)
        self.online_next_a_base = vec(action_dim)
        self.online_done = torch.zeros((max_size,), dtype=torch.bool, device=storage_device)

        self.online_ptr = 0
        self.online_size = 0

        # Offline buffer (filled once via load_offline_trajectories, never evicted)
        self._offline_initialised = False
        self.offline_size = 0
        self.offline_rgb = self.offline_proprio = self.offline_a_base = None
        self.offline_a_delta = self.offline_a_total = self.offline_reward = None
        self.offline_next_rgb = self.offline_next_proprio = self.offline_next_a_base = None
        self.offline_done = self.offline_mc_return = None

        log.info(
            f"PLDReplayBuffer initialised | online_max={max_size}, n_cam={n_cam}, "
            f"image_hw={image_hw}, proprio_dim={proprio_dim}, action_dim={action_dim}"
        )

    # ---------------- adding ----------------

    def _to_storage_tensor(self, x, dtype):
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            t = x.detach()
        else:
            t = torch.tensor(x)
        return t.to(dtype=dtype, device=self.storage_device)

    def add_online(
        self,
        rgb,
        proprio,
        a_base,
        a_delta,
        a_total,
        reward,
        next_rgb,
        next_proprio,
        next_a_base,
        done,
    ):
        """Add a batch of n_envs transitions to the online buffer."""
        rgb_t = self._to_storage_tensor(rgb, torch.uint8)
        proprio_t = self._to_storage_tensor(proprio, torch.float32)
        a_base_t = self._to_storage_tensor(a_base, torch.float32)
        a_delta_t = self._to_storage_tensor(a_delta, torch.float32)
        a_total_t = self._to_storage_tensor(a_total, torch.float32)
        reward_t = self._to_storage_tensor(reward, torch.float32).view(-1)
        next_rgb_t = self._to_storage_tensor(next_rgb, torch.uint8)
        next_proprio_t = self._to_storage_tensor(next_proprio, torch.float32)
        next_a_base_t = self._to_storage_tensor(next_a_base, torch.float32)
        done_t = self._to_storage_tensor(done, torch.bool).view(-1)

        n = rgb_t.shape[0]
        for i in range(n):
            idx = self.online_ptr % self.max_size
            self.online_rgb[idx] = rgb_t[i]
            self.online_proprio[idx] = proprio_t[i]
            self.online_a_base[idx] = a_base_t[i]
            self.online_a_delta[idx] = a_delta_t[i]
            self.online_a_total[idx] = a_total_t[i]
            self.online_reward[idx] = reward_t[i]
            self.online_next_rgb[idx] = next_rgb_t[i]
            self.online_next_proprio[idx] = next_proprio_t[i]
            self.online_next_a_base[idx] = next_a_base_t[i]
            self.online_done[idx] = done_t[i]
            self.online_ptr += 1
            self.online_size = min(self.online_size + 1, self.max_size)

    def load_offline_trajectories(
        self, trajectories: list[dict[str, np.ndarray]]
    ) -> None:
        """Ingest successful demonstration trajectories into the offline buffer
        (one-shot; the offline buffer is filled once and never evicted).

        For each episode it computes a per-step Monte-Carlo return (the Cal-QL
        calibration floor) via a backward discounted sum that resets on done,
        then flattens the episode into per-step transitions t -> t+1. The final
        step (done=True) has no successor and is dropped, so a T-step episode
        yields T-1 transitions.

        Args:
            trajectories: list of successful episodes, one dict per episode (the
                output of collect_offline_trajectories). Each dict holds these
                arrays, stacked over that episode's T steps:

                    key      | shape               | dtype   | meaning
                    -------- | ------------------- | ------- | -------------------------
                    rgb      | (T, N_cam, H, W, 3) | uint8   | camera frames at step t
                    proprio  | (T, proprio_dim)    | float32 | robot state at step t
                    a_base   | (T, action_dim)     | float32 | normalized base action run
                    reward   | (T,)                | float32 | env reward at step t
                    done     | (T,)                | bool    | terminal flag, True only at t = T-1

                T varies per episode and must be >= 2.

        Returns:
            None. Populates the offline_* storage tensors, sets offline_size to
            the total transition count, and flips _offline_initialised to True.
        """
        offline_rgb = []
        offline_proprio = []
        offline_a_base = []
        offline_a_delta = []
        offline_a_total = []
        offline_reward = []
        offline_next_rgb = []
        offline_next_proprio = []
        offline_next_a_base = []
        offline_done = []
        offline_mc_return = []

        for traj in trajectories:
            T = traj["rgb"].shape[0]
            assert T >= 2, "Offline trajectory must have at least 2 steps"

            rgb = np.asarray(traj["rgb"])
            proprio = np.asarray(traj["proprio"])
            a_base = np.asarray(traj["a_base"])
            reward = np.asarray(traj["reward"], dtype=np.float32)
            done = np.asarray(traj["done"], dtype=bool)

            # MC return per step (discounted future reward from step t to end)
            mc = np.zeros(T, dtype=np.float32)
            running = 0.0
            for t in reversed(range(T)):
                running = reward[t] + (self.gamma * running if not done[t] else 0.0)
                mc[t] = running

            # transitions: t -> t+1 for t in [0, T-2]; last step (done) has no next
            for t in range(T - 1):
                offline_rgb.append(rgb[t])
                offline_proprio.append(proprio[t])
                offline_a_base.append(a_base[t])
                offline_a_delta.append(np.zeros(self.action_dim, dtype=np.float32))
                offline_a_total.append(a_base[t])  # a_total = a_base since a_delta = 0
                offline_reward.append(reward[t])
                offline_next_rgb.append(rgb[t + 1])
                offline_next_proprio.append(proprio[t + 1])
                offline_next_a_base.append(a_base[t + 1])
                offline_done.append(done[t])
                offline_mc_return.append(mc[t])

        N = len(offline_rgb)
        log.info(f"Loading {N} offline transitions from {len(trajectories)} trajectories")

        self.offline_rgb = torch.from_numpy(np.stack(offline_rgb)).to(self.storage_device)
        self.offline_proprio = torch.from_numpy(np.stack(offline_proprio)).to(self.storage_device)
        self.offline_a_base = torch.from_numpy(np.stack(offline_a_base)).to(self.storage_device)
        self.offline_a_delta = torch.from_numpy(np.stack(offline_a_delta)).to(self.storage_device)
        self.offline_a_total = torch.from_numpy(np.stack(offline_a_total)).to(self.storage_device)
        self.offline_reward = torch.from_numpy(np.asarray(offline_reward, dtype=np.float32)).to(
            self.storage_device
        )
        self.offline_next_rgb = torch.from_numpy(np.stack(offline_next_rgb)).to(self.storage_device)
        self.offline_next_proprio = torch.from_numpy(np.stack(offline_next_proprio)).to(
            self.storage_device
        )
        self.offline_next_a_base = torch.from_numpy(np.stack(offline_next_a_base)).to(
            self.storage_device
        )
        self.offline_done = torch.from_numpy(np.asarray(offline_done, dtype=bool)).to(
            self.storage_device
        )
        self.offline_mc_return = torch.from_numpy(np.asarray(offline_mc_return, dtype=np.float32)).to(
            self.storage_device
        )

        self.offline_size = N
        self._offline_initialised = True

    # ---------------- sampling ----------------

    def _gather(self, source: str, indices: torch.Tensor):
        """Gather a batch from the named source ('online' or 'offline')."""
        prefix = f"{source}_"
        rgb = getattr(self, prefix + "rgb")[indices]
        proprio = getattr(self, prefix + "proprio")[indices]
        a_base = getattr(self, prefix + "a_base")[indices]
        a_delta = getattr(self, prefix + "a_delta")[indices]
        a_total = getattr(self, prefix + "a_total")[indices]
        reward = getattr(self, prefix + "reward")[indices]
        next_rgb = getattr(self, prefix + "next_rgb")[indices]
        next_proprio = getattr(self, prefix + "next_proprio")[indices]
        next_a_base = getattr(self, prefix + "next_a_base")[indices]
        done = getattr(self, prefix + "done")[indices]
        if source == "offline":
            mc_return = self.offline_mc_return[indices]
        else:
            # Online transitions don't have calibrated MC returns; use 0
            mc_return = torch.zeros_like(reward)
        return {
            "rgb": rgb,
            "proprio": proprio,
            "a_base": a_base,
            "a_delta": a_delta,
            "a_total": a_total,
            "reward": reward,
            "next_rgb": next_rgb,
            "next_proprio": next_proprio,
            "next_a_base": next_a_base,
            "done": done,
            "mc_return": mc_return,
        }

    def sample(self, batch_size: int, expert_ratio: float = 0.5):
        """Sample a 50/50 mixed batch from online + offline buffers."""
        n_offline = int(batch_size * expert_ratio)
        n_online = batch_size - n_offline

        # Edge cases: not enough offline or online data
        if not self._offline_initialised or self.offline_size == 0:
            n_offline = 0
            n_online = batch_size
        if self.online_size == 0:
            n_offline = batch_size
            n_online = 0

        batches = []
        if n_offline > 0:
            idx = torch.randint(0, self.offline_size, (n_offline,), device=self.storage_device)
            batches.append(self._gather("offline", idx))
        if n_online > 0:
            idx = torch.randint(0, self.online_size, (n_online,), device=self.storage_device)
            batches.append(self._gather("online", idx))

        # Concatenate
        out = {k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0].keys()}

        # Move to sample_device
        if self.sample_device != self.storage_device:
            out = {k: v.to(self.sample_device, non_blocking=True) for k, v in out.items()}
        return out

    def sample_offline_only(self, batch_size: int):
        if not self._offline_initialised or self.offline_size == 0:
            raise RuntimeError("Offline buffer not initialised")
        idx = torch.randint(0, self.offline_size, (batch_size,), device=self.storage_device)
        out = self._gather("offline", idx)
        if self.sample_device != self.storage_device:
            out = {k: v.to(self.sample_device, non_blocking=True) for k, v in out.items()}
        return out

    def __len__(self):
        return self.online_size + (self.offline_size if self._offline_initialised else 0)
