#!/usr/bin/env python3
"""Standalone evaluation of a trained PLD residual policy on RoboCasa.

Evaluates ``base GR00T + learned residual actor`` for N episodes on the robocasa
target task, reusing ``agent.train_residual_rl.TrainResidualRL.evaluate()`` for
the rollout logic and the GR00T-load pattern from the trainer / eval_gr00t.

Respects the 4-env hard cap (``util.env_pool.MAX_ROBOCASA_ENVS``): pass
``--n_envs <= 4``. There is exactly ONE shared env pool.

Run inside the ``robocasa`` conda env, from the project root so the Hydra-style
config path resolves, e.g.:

  python script/eval_residual_policy.py \\
    --config cfg/robocasa/finetune/CloseToasterOvenDoor/calql_sac.yaml \\
    --checkpoint /path/to/checkpoint/best.pt \\
    --n_episodes 100 --n_envs 4
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Register resolvers used by the configs (mirrors script/run.py).
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
OmegaConf.register_new_resolver("round_down", math.floor, replace=True)
# Hydra normally provides ${now:...}; register it here since we run outside Hydra.
OmegaConf.register_new_resolver(
    "now", lambda fmt: datetime.now().strftime(fmt), replace=True
)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

from util.env_pool import MAX_ROBOCASA_ENVS  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("eval_residual_policy")


def _load_config(config_path: str):
    """Load a config the same way Hydra would resolve its defaults list.

    The variant configs use a Hydra ``defaults:`` list (``- _base_residual_rl``),
    which OmegaConf.load does not expand. We merge the referenced base config(s)
    from the same directory manually so this script is runnable without Hydra.
    """
    config_path = Path(config_path).resolve()
    cfg = OmegaConf.load(config_path)

    merged = OmegaConf.create({})
    defaults = cfg.get("defaults", []) if "defaults" in cfg else []
    for entry in defaults:
        if entry == "_self_":
            merged = OmegaConf.merge(merged, cfg)
        elif isinstance(entry, str):
            base_path = config_path.parent / f"{entry}.yaml"
            base_cfg = OmegaConf.load(base_path)
            merged = OmegaConf.merge(merged, base_cfg)
        else:
            # mapping form (e.g. {group: option}) — not used by these configs
            raise NotImplementedError(f"Unsupported defaults entry: {entry}")
    if not defaults:
        merged = cfg
    elif "_self_" not in defaults:
        # Hydra applies the primary config last when _self_ is absent.
        merged = OmegaConf.merge(merged, cfg)
    # drop the defaults key so downstream code doesn't choke on it
    if "defaults" in merged:
        del merged["defaults"]
    return merged


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="Path to the variant yaml config")
    p.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint .pt")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--n_envs", type=int, default=4, help=f"<= {MAX_ROBOCASA_ENVS}")
    p.add_argument("--device", default=None, help="Override cfg.device (e.g. cuda:0)")
    p.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use the deterministic residual mode (default true).",
    )
    p.add_argument(
        "--zero_residual",
        action="store_true",
        default=False,
        help="Force residual=0 to evaluate the frozen base GR00T policy alone.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    assert args.n_envs <= MAX_ROBOCASA_ENVS, (
        f"--n_envs={args.n_envs} exceeds the hard cap {MAX_ROBOCASA_ENVS}."
    )

    cfg = _load_config(args.config)

    # Apply CLI overrides. Eval reuses the single pool, so n_envs is the pool size.
    cfg.env.n_envs = args.n_envs
    if args.device is not None:
        cfg.device = args.device
    # Disable wandb for standalone eval.
    cfg.wandb = None
    # We only run eval; skip offline/pretrain/warmup work by zeroing those knobs
    # (run() is not called — we call evaluate() directly — but keep cfg coherent).
    cfg.run_eval = True

    # Build the trainer harness WITHOUT running phases; this constructs the GR00T
    # provider, normalizer, collector, env pool, algorithm graph and buffer.
    from agent.train_residual_rl import TrainResidualRL

    trainer = TrainResidualRL(cfg)

    # Load checkpoint into the algorithm (incl. log_alpha).
    log.info(f"Loading checkpoint from {args.checkpoint}")
    data = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
    state = data["model"] if isinstance(data, dict) and "model" in data else data
    trainer.algorithm.load_state_dict(state)
    trainer.algorithm.eval()

    log.info(
        f"Evaluating residual policy for {args.n_episodes} episodes on "
        f"{cfg.env.name} with {args.n_envs} envs (deterministic={args.deterministic})"
    )
    sr = trainer.evaluate(
        args.n_episodes, deterministic=args.deterministic,
        zero_residual=args.zero_residual,
    )
    log.info("=" * 60)
    log.info(f"FINAL success rate ({args.n_episodes} episodes): {sr:.4f} ({sr*100:.1f}%)")
    log.info("=" * 60)
    print(f"success_rate={sr:.4f}")

    trainer.env_pool.close()


if __name__ == "__main__":
    main()
