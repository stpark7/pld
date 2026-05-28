"""
Parent training agent class for the PLD residual-RL project.

Slimmed from dice-rl ``agent/finetune/train_agent.py``: keeps seeding /
determinism setup, wandb init, checkpoint I/O, and config plumbing, but the
env creation is delegated to the single shared 4-env pool (``util.env_pool``)
instead of building separate training + eval vectorized envs.
"""

import os
import random
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
import wandb

from util.env_pool import EnvPool

log = logging.getLogger(__name__)


class TrainAgent:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # CUDA deterministic settings
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

        # Wandb
        self.use_wandb = cfg.get("wandb", None) is not None
        try:
            run_dir = Path(HydraConfig.get().runtime.output_dir)
            run_name = run_dir.name
        except Exception:
            run_name = cfg.get("name", "pld_residual_rl")
        if self.use_wandb:
            wandb.init(
                project=cfg.wandb.project,
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        # Single shared env pool (NO separate eval pool — eval reuses this one).
        self.env_name = cfg.env.name
        self.env_pool = EnvPool(cfg, seed=self.seed)
        self.venv = self.env_pool.venv
        self.n_envs = self.env_pool.n_envs
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        self.max_episode_steps = cfg.env.max_episode_steps

        # Batch size for gradient update
        self.batch_size: int = cfg.train.batch_size

        # Logging / checkpoints
        self.logdir = cfg.logdir
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.itr = 0

    def run(self):
        pass

    # ------------------------------------------------------------ checkpoint I/O

    def save_model(self):
        """Save the algorithm state (no ema)."""
        data = {
            "itr": self.itr,
            "model": self.algorithm.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
        torch.save(data, savepath)
        log.info(f"Saved model to {savepath}")

    def load(self, itr):
        """Load the algorithm state from disk."""
        loadpath = os.path.join(self.checkpoint_dir, f"state_{itr}.pt")
        data = torch.load(loadpath, weights_only=False)
        self.itr = data["itr"]
        self.algorithm.load_state_dict(data["model"])
