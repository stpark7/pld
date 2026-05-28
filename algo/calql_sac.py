"""
Cal-QL + SAC residual-RL algorithm plugin (PLD Stage 1).

Ported from dice-rl ``model/rl/pld_stage1.py`` (PLDStage1Model losses) plus the
optimizer wiring / log_alpha / pretrain-step / online-update loop that used to
live in ``agent/finetune/train_pld_stage1.py``. All of that is now behind the
``ResidualRLAlgorithm`` contract so the orchestrator is algorithm-agnostic.

Loss math (UNCHANGED from the working dice-rl implementation):
  - Cal-QL TD loss with N action samples for max-Q next action.
  - Conservative term: logsumexp over random + policy actions.
  - Calibration floor: q_pi floored at the offline MC return.
  - SAC actor loss (-min(Q1,Q2) + alpha * logp).
  - SAC temperature loss (auto-tuned alpha).
  - Polyak target update.

Diagnostics (known-risk fix #1): the critic update additionally logs q_data
mean, the calibration floor (mc_return) stats, the fraction of samples where the
floor binds (q_pi < mc_return), and the cql_diff terms, so the conservative /
calibration terms can be verified active under sparse 0/1 rewards.
"""

from __future__ import annotations

import math
import logging
from copy import deepcopy

import einops
import numpy as np
import torch
import torch.nn as nn

from algo.base import ResidualRLAlgorithm

log = logging.getLogger(__name__)


class PLDCalQLSAC(ResidualRLAlgorithm):
    def __init__(
        self,
        encoder: nn.Module,
        actor: nn.Module,
        critic: nn.Module,
        action_dim: int,
        gamma: float = 0.99,
        polyak_tau: float = 0.005,
        cql_min_q_weight: float = 0.5,
        cql_n_actions: int = 10,
        cql_n_random: int = 10,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        use_calibration: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self._gamma = gamma
        self.polyak_tau = polyak_tau
        self.cql_min_q_weight = cql_min_q_weight
        self.cql_n_actions = cql_n_actions
        self.cql_n_random = cql_n_random
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self.use_calibration = use_calibration

        self.encoder = encoder.to(device)
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.target_critic = deepcopy(critic).to(device)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # Temperature / optimizers are wired up in build_optimizers(cfg).
        self.log_alpha = None
        self.encoder_critic_optim = None
        self.actor_optim = None
        self.alpha_optim = None
        self.max_grad_norm = 1.0
        self.target_entropy = -action_dim / 2.0

        log.info(
            f"PLDCalQLSAC | encoder={sum(p.numel() for p in encoder.parameters())} params, "
            f"actor={sum(p.numel() for p in actor.parameters())} params, "
            f"critic={sum(p.numel() for p in critic.parameters())} params"
        )

    # ============================================================ contract: properties

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def needs_pretrain(self) -> bool:
        return True

    @property
    def needs_mc_return(self) -> bool:
        return self.use_calibration

    # ============================================================ contract: optimizers

    def build_optimizers(self, cfg) -> None:
        """Wire up encoder+critic / actor / temperature optimizers and log_alpha.

        Moved out of the trainer so the orchestrator stays algo-agnostic.
        """
        lr = float(cfg.train.lr)
        self.max_grad_norm = float(cfg.train.get("max_grad_norm", 1.0))
        self.target_entropy = float(
            cfg.train.get("target_entropy", -self.action_dim / 2.0)
        )

        # Encoder and critic share the critic loss; actor and log_alpha separate.
        self.encoder_critic_optim = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=lr)
        self.log_alpha = torch.tensor(
            float(cfg.train.get("init_log_alpha", 0.0)),
            requires_grad=True,
            device=self.device,
        )
        self.alpha_optim = torch.optim.AdamW([self.log_alpha], lr=lr)

    def _alpha(self) -> float:
        return self.log_alpha.exp().item()

    # ============================================================ utility (verbatim)

    def _state_cond(self, feat: torch.Tensor, proprio: torch.Tensor) -> dict:
        """Build the cond dict expected by CriticObsAct (state-only critic interface)."""
        x = torch.cat([feat, proprio], dim=-1).unsqueeze(1)  # (B, 1, feat+proprio)
        return {"state": x}

    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.encoder(rgb)

    # ============================================================ contract: inference

    @torch.no_grad()
    def get_residual_action(
        self,
        rgb: torch.Tensor,
        proprio: torch.Tensor,
        a_base: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        feat = self.encoder(rgb)
        a_delta = self.actor(feat, proprio, a_base, deterministic=deterministic)
        return a_delta

    def select_action(self, rgb, proprio, a_base, deterministic: bool = True):
        """Contract entry point: residual action for the orchestrator/eval.

        During online collection the trainer wants a stochastic (mean-shifted but
        non-reparameterized) sample; during eval it wants the deterministic mode.
        Both are handled here based on ``deterministic``.
        """
        self.eval()
        with torch.no_grad():
            feat = self.encoder(rgb)
            a_delta = self.actor(
                feat,
                proprio,
                a_base,
                deterministic=deterministic,
                reparameterize=False,
                get_logprob=False,
            )
        self.train()
        return a_delta

    # ============================================================ critic loss (Cal-QL, verbatim + diagnostics)

    def loss_critic(
        self,
        batch: dict,
        alpha: float,
        gamma: float = None,
    ):
        """
        batch keys: rgb, proprio, a_base, a_total, reward, next_rgb, next_proprio,
                    next_a_base, done, mc_return.
        """
        if gamma is None:
            gamma = self._gamma

        rgb = batch["rgb"]
        proprio = batch["proprio"]
        a_total = batch["a_total"]
        reward = batch["reward"]
        next_rgb = batch["next_rgb"]
        next_proprio = batch["next_proprio"]
        next_a_base = batch["next_a_base"]
        done = batch["done"].float()
        mc_return = batch["mc_return"]

        B = a_total.shape[0]

        # Encoder forward — gradient flows from critic loss only
        feat = self.encoder(rgb)
        with torch.no_grad():
            next_feat = self.encoder(next_rgb)

        # Current Q
        cond = self._state_cond(feat, proprio)
        q_data1, q_data2 = self.critic(cond, a_total)

        # TD target: sample N next actions, take max-Q
        with torch.no_grad():
            n_act = self.cql_n_actions
            next_feat_rep = next_feat.repeat_interleave(n_act, dim=0)
            next_proprio_rep = next_proprio.repeat_interleave(n_act, dim=0)
            next_a_base_rep = next_a_base.repeat_interleave(n_act, dim=0)

            next_a_delta_rep, next_logp_rep = self.actor(
                next_feat_rep,
                next_proprio_rep,
                next_a_base_rep,
                deterministic=False,
                reparameterize=False,
                get_logprob=True,
            )
            next_a_total_rep = next_a_base_rep + next_a_delta_rep

            next_cond_rep = self._state_cond(next_feat_rep, next_proprio_rep)
            next_q1_rep, next_q2_rep = self.target_critic(next_cond_rep, next_a_total_rep)
            next_q_rep = torch.min(next_q1_rep, next_q2_rep)

            next_q_rep = next_q_rep.view(B, n_act)
            next_logp_rep = next_logp_rep.view(B, n_act)

            max_idx = torch.argmax(next_q_rep, dim=1)
            arange = torch.arange(B, device=next_q_rep.device)
            next_q = next_q_rep[arange, max_idx]
            next_logp = next_logp_rep[arange, max_idx]

            target_q = reward + gamma * (1.0 - done) * (next_q - alpha * next_logp)

        td_loss = nn.functional.mse_loss(q_data1, target_q) + nn.functional.mse_loss(
            q_data2, target_q
        )

        # ---- Conservative term ----
        n_rand = self.cql_n_random
        # Random actions uniform in [-1, 1]
        rand_actions = torch.rand(
            (B, n_rand, self.action_dim), device=feat.device
        ) * 2.0 - 1.0
        log_rand = -self.action_dim * math.log(2.0)  # log(0.5) per dim — uniform on [-1,1]

        feat_rep = feat.repeat_interleave(n_rand, dim=0)
        proprio_rep = proprio.repeat_interleave(n_rand, dim=0)
        cond_rep = self._state_cond(feat_rep, proprio_rep)
        rand_actions_flat = einops.rearrange(rand_actions, "B N A -> (B N) A")
        q_rand_1, q_rand_2 = self.critic(cond_rep, rand_actions_flat)
        q_rand_1 = q_rand_1.view(B, n_rand) - log_rand
        q_rand_2 = q_rand_2.view(B, n_rand) - log_rand

        # Policy actions on current state (a_total_pi) — no gradient through actor here
        with torch.no_grad():
            a_delta_pi, log_pi = self.actor(
                feat, proprio, batch["a_base"],
                deterministic=False, reparameterize=False, get_logprob=True,
            )
            a_total_pi = batch["a_base"] + a_delta_pi
        q_pi_1, q_pi_2 = self.critic(cond, a_total_pi)
        q_pi_1 = q_pi_1 - log_pi
        q_pi_2 = q_pi_2 - log_pi

        # ---- Calibration diagnostics (known-risk fix #1) ----
        # Before flooring, record how often / how much the MC-return floor binds.
        with torch.no_grad():
            floor_binds_1 = (q_pi_1 < mc_return).float().mean()
            floor_binds_2 = (q_pi_2 < mc_return).float().mean()
            floor_binds_frac = 0.5 * (floor_binds_1 + floor_binds_2)
            q_pi_pre_floor_mean = 0.5 * (q_pi_1.mean() + q_pi_2.mean())

        # Calibration: floor q_pi by MC return (only meaningful on offline samples;
        # online samples have mc_return = 0 which is a non-active floor under sparse rewards)
        if self.use_calibration:
            q_pi_1 = torch.max(q_pi_1, mc_return)[:, None]
            q_pi_2 = torch.max(q_pi_2, mc_return)[:, None]
        else:
            q_pi_1 = q_pi_1[:, None]
            q_pi_2 = q_pi_2[:, None]

        cat_q_1 = torch.cat([q_rand_1, q_pi_1], dim=-1)
        cat_q_2 = torch.cat([q_rand_2, q_pi_2], dim=-1)
        cql_qf1_ood = torch.logsumexp(cat_q_1, dim=-1)
        cql_qf2_ood = torch.logsumexp(cat_q_2, dim=-1)

        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q_data1, min=self.cql_clip_diff_min, max=self.cql_clip_diff_max
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q_data2, min=self.cql_clip_diff_min, max=self.cql_clip_diff_max
        ).mean()
        cql_loss = self.cql_min_q_weight * (cql_qf1_diff + cql_qf2_diff)

        loss = td_loss + cql_loss

        info = {
            "td_loss": td_loss.detach(),
            "cql_loss": cql_loss.detach(),
            "q_data1_mean": q_data1.detach().mean(),
            "q_data2_mean": q_data2.detach().mean(),
            "target_q_mean": target_q.detach().mean(),
            "cql_qf1_diff": cql_qf1_diff.detach(),
            "cql_qf2_diff": cql_qf2_diff.detach(),
            # ---- calibration diagnostics (fix #1) ----
            "mc_return_mean": mc_return.detach().mean(),
            "mc_return_max": mc_return.detach().max(),
            "mc_return_nonzero_frac": (mc_return.detach() > 0).float().mean(),
            "q_pi_pre_floor_mean": q_pi_pre_floor_mean.detach(),
            "calib_floor_binds_frac": floor_binds_frac.detach(),
        }
        return loss, info

    # ============================================================ actor loss (SAC, verbatim)

    def loss_actor(self, batch: dict, alpha: float):
        rgb = batch["rgb"]
        proprio = batch["proprio"]
        a_base = batch["a_base"]

        with torch.no_grad():
            feat = self.encoder(rgb)  # actor doesn't update encoder

        a_delta, log_p = self.actor(
            feat, proprio, a_base,
            deterministic=False, reparameterize=True, get_logprob=True,
        )
        a_total = a_base + a_delta

        cond = self._state_cond(feat, proprio)
        q1, q2 = self.critic(cond, a_total)
        q = torch.min(q1, q2)
        loss = (alpha * log_p - q).mean()

        info = {
            "actor_loss": loss.detach(),
            "log_p_mean": log_p.detach().mean(),
            "q_pi_mean": q.detach().mean(),
        }
        return loss, info

    # ============================================================ temperature loss (SAC alpha, verbatim)

    def loss_temperature(self, batch: dict, log_alpha: torch.Tensor, target_entropy: float):
        rgb = batch["rgb"]
        proprio = batch["proprio"]
        a_base = batch["a_base"]
        with torch.no_grad():
            feat = self.encoder(rgb)
            _, log_p = self.actor(
                feat, proprio, a_base,
                deterministic=False, reparameterize=False, get_logprob=True,
            )
        loss = -(log_alpha * (log_p + target_entropy).detach()).mean()
        return loss, {"alpha": log_alpha.exp().detach(), "log_p_mean": log_p.detach().mean()}

    # ============================================================ target update (verbatim)

    def polyak_update(self, tau: float = None):
        if tau is None:
            tau = self.polyak_tau
        with torch.no_grad():
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    # ============================================================ contract: critic pretrain

    def pretrain_critic_step(self, offline_batch) -> dict:
        """One Cal-QL critic-init step on an offline batch (encoder updates too).

        Mirrors the per-step body of the old ``train_pld_stage1.pretrain_critic``.
        """
        self.train()
        critic_loss, info = self.loss_critic(offline_batch, alpha=self._alpha())
        self.encoder_critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.critic.parameters()),
            self.max_grad_norm,
        )
        self.encoder_critic_optim.step()
        self.polyak_update()
        out = {"critic_loss": critic_loss.detach()}
        out.update(info)
        return out

    # ============================================================ contract: online update

    def update(self, batch, utd: int) -> dict:
        """One online update cycle.

        UTD critic updates (each on a fresh-style use of ``batch``, mirroring the
        original loop which reused the sampled batch across UTD critic steps),
        then one actor update, one temperature update, with polyak after each
        critic update.

        NOTE: faithfully mirrors ``train_pld_stage1.train_loop`` — the same
        ``batch`` is reused for all UTD critic steps and the actor/temp steps.
        """
        self.train()
        alpha = self._alpha()

        # ---- UTD critic updates (encoder + critic) ----
        for _ in range(utd):
            critic_loss, info_c = self.loss_critic(batch, alpha=self._alpha())
            self.encoder_critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm,
            )
            self.encoder_critic_optim.step()
            self.polyak_update()

        # ---- Actor update ----
        actor_loss, info_a = self.loss_actor(batch, alpha=self._alpha())
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        # ---- Temperature update ----
        tloss, info_t = self.loss_temperature(batch, self.log_alpha, self.target_entropy)
        self.alpha_optim.zero_grad()
        tloss.backward()
        self.alpha_optim.step()

        out = {
            "critic_loss": critic_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "temp_loss": tloss.detach(),
            "alpha": self._alpha(),
        }
        out.update(info_c)
        out.update(info_a)
        out.update(info_t)
        return out

    # ============================================================ contract: state_dict (incl. log_alpha)

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if self.log_alpha is not None:
            sd["log_alpha"] = self.log_alpha.detach()
        return sd

    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict = dict(state_dict)
        log_alpha = state_dict.pop("log_alpha", None)
        result = super().load_state_dict(state_dict, strict=strict)
        if log_alpha is not None:
            if self.log_alpha is None:
                self.log_alpha = torch.tensor(
                    float(log_alpha), requires_grad=True, device=self.device
                )
            else:
                with torch.no_grad():
                    self.log_alpha.copy_(log_alpha.to(self.log_alpha.device))
        return result
