"""
Pluggable residual-RL algorithm contract.

``ResidualRLAlgorithm`` is the single plug point that decouples the
algorithm-specific losses / critic-init / optimizer wiring from the
env / buffer / collector / eval / train-loop orchestration.

Swapping the algorithm = a new ``algo/<x>.py`` implementing this contract +
a new cfg ``_target_``, with NO change to the orchestrator
(``agent.train_residual_rl.TrainResidualRL``), the env pool, the replay buffer,
the collector, or the eval logic.

Contract (everything the orchestrator calls):
  - ``build_optimizers(cfg)``        : wire up optimizers + temperature params.
  - ``select_action(rgb, proprio, a_base, deterministic) -> a_delta`` : inference.
  - ``pretrain_critic_step(offline_batch) -> info``                   : critic init.
  - ``update(batch, utd) -> info``                                    : one online cycle.
  - ``state_dict()`` / ``load_state_dict()``  (incl. any temperature params).
  - properties: ``gamma``, ``needs_pretrain``, ``needs_mc_return``.

The algorithm owns the encoder / actor / critic submodules (it is an
``nn.Module``), so a single ``state_dict()`` round-trips the whole graph plus
any extra learnable scalars (e.g. ``log_alpha``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn


class ResidualRLAlgorithm(ABC, nn.Module):
    """Abstract base class for a pluggable residual-RL algorithm.

    Concrete subclasses (e.g. ``PLDCalQLSAC``) own the actor/critic/encoder and
    all loss math; the orchestrator only ever touches the methods/properties
    declared here.
    """

    def __init__(self):
        super().__init__()

    # ---------------- optimizer wiring ----------------

    @abstractmethod
    def build_optimizers(self, cfg) -> None:
        """Create optimizers (and any temperature parameters) from ``cfg``.

        Called once by the orchestrator after instantiation. Stores optimizers
        as attributes on the algorithm so that ``pretrain_critic_step`` and
        ``update`` can use them. Keeps the orchestrator algo-agnostic (no
        log_alpha / optimizer wiring lives in the trainer).
        """
        raise NotImplementedError

    # ---------------- inference ----------------

    @abstractmethod
    def select_action(self, rgb, proprio, a_base, deterministic: bool = True):
        """Return the residual action ``a_delta`` for the given observation.

        Args:
            rgb:      (B, N_cam, H, W, 3) image tensor.
            proprio:  (B, proprio_dim) proprio tensor.
            a_base:   (B, action_dim) normalized base action.
            deterministic: if True, return the mode of the residual policy.

        Returns:
            a_delta: (B, action_dim) residual action tensor.
        """
        raise NotImplementedError

    # ---------------- critic pretraining ----------------

    @abstractmethod
    def pretrain_critic_step(self, offline_batch) -> dict:
        """Run one critic-init update on an offline batch; return a log-info dict.

        Variants that need no critic pretrain (``needs_pretrain == False``) may
        leave this as a no-op (the orchestrator gates on ``needs_pretrain``).
        """
        raise NotImplementedError

    # ---------------- online update ----------------

    @abstractmethod
    def update(self, batch, utd: int) -> dict:
        """Run one online update cycle (UTD critic updates + actor + temperature
        + target polyak) and return a log-info dict.

        Args:
            batch: a sampled minibatch dict (50/50 online+offline).
            utd:   update-to-data ratio (critic updates per actor update).
        """
        raise NotImplementedError

    # ---------------- properties ----------------

    @property
    @abstractmethod
    def gamma(self) -> float:
        """Discount factor (used by the replay buffer for MC-return computation)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def needs_pretrain(self) -> bool:
        """Whether the orchestrator should run the critic-pretrain phase."""
        raise NotImplementedError

    @property
    @abstractmethod
    def needs_mc_return(self) -> bool:
        """Whether the algorithm consumes offline MC returns (e.g. Cal-QL floor)."""
        raise NotImplementedError
