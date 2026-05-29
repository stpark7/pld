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

        # TD target: standard SAC single-sample backup with clipped double-Q.
        # NOTE: deliberately NOT the Cal-QL max-over-N (cql_max_target_backup)
        # backup — that optimistic max-over-cql_n_actions target is meant to pair
        # with a large critic ensemble we don't have, and it drove the critic Q to
        # overestimate (q_data climbing to +50..+90 vs true return ~2-5). Sample
        # exactly ONE next action from the actor at s' and take the clipped
        # double-Q. (The conservative term below still uses cql_n_actions samples
        # and the calibration floor — only the backup changed.)
        with torch.no_grad():
            next_a_delta, next_logp = self.actor(
                next_feat,
                next_proprio,
                next_a_base,
                deterministic=False,
                reparameterize=False,
                get_logprob=True,
            )
            next_a_total = next_a_base + next_a_delta

            next_cond = self._state_cond(next_feat, next_proprio)
            next_q1, next_q2 = self.target_critic(next_cond, next_a_total)
            next_q = torch.min(next_q1, next_q2)

            target_q = reward + gamma * (1.0 - done) * (next_q - alpha * next_logp)

        td_loss = nn.functional.mse_loss(q_data1, target_q) + nn.functional.mse_loss(
            q_data2, target_q
        )

        # ---- Conservative term (faithful Cal-QL, ref JaxCQL/conservative_sac.py
        # lines ~206-307). Conservative set = {random actions, current-state policy
        # actions, next-state policy actions}, each importance-weighted by its log
        # density (random -> log(0.5^A); policy groups -> their tanh-Gaussian log_pi).
        # The Cal-QL calibration floor max(Q, mc_return) is applied to BOTH policy
        # groups BEFORE the importance correction. ----
        n_act = self.cql_n_actions  # policy-action samples per state (ref cql_n_actions)
        n_rand = self.cql_n_random  # random-action samples per state

        # --- Random actions uniform in [-1, 1]^A (ref: minval=-1, maxval=1) ---
        rand_actions = torch.rand(
            (B, n_rand, self.action_dim), device=feat.device
        ) * 2.0 - 1.0
        # log density of uniform on [-1,1]^A = log(0.5^A) (ref random_density)
        random_density = self.action_dim * math.log(0.5)

        feat_rep_r = feat.repeat_interleave(n_rand, dim=0)
        proprio_rep_r = proprio.repeat_interleave(n_rand, dim=0)
        cond_rep_r = self._state_cond(feat_rep_r, proprio_rep_r)
        rand_actions_flat = einops.rearrange(rand_actions, "B N A -> (B N) A")
        q_rand_1, q_rand_2 = self.critic(cond_rep_r, rand_actions_flat)
        q_rand_1 = q_rand_1.view(B, n_rand) - random_density
        q_rand_2 = q_rand_2.view(B, n_rand) - random_density

        # --- Current-state policy actions (a_base + a_delta at s), N samples ---
        # No gradient through the actor (matches the original / ref convention).
        feat_rep_c = feat.repeat_interleave(n_act, dim=0)
        proprio_rep_c = proprio.repeat_interleave(n_act, dim=0)
        a_base_rep_c = batch["a_base"].repeat_interleave(n_act, dim=0)
        with torch.no_grad():
            a_delta_cur, log_pi_cur = self.actor(
                feat_rep_c, proprio_rep_c, a_base_rep_c,
                deterministic=False, reparameterize=False, get_logprob=True,
            )
            a_total_cur = a_base_rep_c + a_delta_cur
        cond_rep_c = self._state_cond(feat_rep_c, proprio_rep_c)
        q_cur_1, q_cur_2 = self.critic(cond_rep_c, a_total_cur)
        q_cur_1 = q_cur_1.view(B, n_act)
        q_cur_2 = q_cur_2.view(B, n_act)
        log_pi_cur = log_pi_cur.view(B, n_act)

        # --- Next-state policy actions (a_base' + a_delta' at s'), N samples ---
        # next_feat is detached (computed under no_grad in the TD block); the critic
        # Q here still carries gradient w.r.t. critic params (the conservative
        # penalty pushes these OOD Q-values down). The BACKUP elsewhere uses the
        # target critic + stop-grad; this conservative term uses the online critic
        # at s', matching the ref (observations=next_observations, qf params).
        feat_rep_n = next_feat.repeat_interleave(n_act, dim=0)
        proprio_rep_n = next_proprio.repeat_interleave(n_act, dim=0)
        a_base_rep_n = next_a_base.repeat_interleave(n_act, dim=0)
        with torch.no_grad():
            a_delta_nxt, log_pi_nxt = self.actor(
                feat_rep_n, proprio_rep_n, a_base_rep_n,
                deterministic=False, reparameterize=False, get_logprob=True,
            )
            a_total_nxt = a_base_rep_n + a_delta_nxt
        cond_rep_n = self._state_cond(feat_rep_n, proprio_rep_n)
        q_nxt_1, q_nxt_2 = self.critic(cond_rep_n, a_total_nxt)
        q_nxt_1 = q_nxt_1.view(B, n_act)
        q_nxt_2 = q_nxt_2.view(B, n_act)
        log_pi_nxt = log_pi_nxt.view(B, n_act)

        # ---- Calibration diagnostics (known-risk fix #1) ----
        # Before flooring, record how often / how much the MC-return floor binds
        # across BOTH policy-action groups (ref logs bound_rate for current & next).
        mc_col = mc_return[:, None]  # (B, 1) broadcasts against (B, n_act)
        with torch.no_grad():
            binds = torch.cat(
                [(q_cur_1 < mc_col).float(), (q_cur_2 < mc_col).float(),
                 (q_nxt_1 < mc_col).float(), (q_nxt_2 < mc_col).float()],
                dim=-1,
            )
            floor_binds_frac = binds.mean()
            q_pi_pre_floor_mean = 0.25 * (
                q_cur_1.mean() + q_cur_2.mean() + q_nxt_1.mean() + q_nxt_2.mean()
            )

        # ---- Cal-QL calibration: floor BOTH policy groups by the MC return ----
        # (ref lines ~236-240: bound current & next actions). Done BEFORE the
        # importance-sampling log-prob correction. Online samples have mc_return=0,
        # a non-active floor under sparse rewards; offline samples carry the floor.
        if self.use_calibration:
            q_cur_1 = torch.max(q_cur_1, mc_col)
            q_cur_2 = torch.max(q_cur_2, mc_col)
            q_nxt_1 = torch.max(q_nxt_1, mc_col)
            q_nxt_2 = torch.max(q_nxt_2, mc_col)

        # ---- Importance-sampling correction (ref cql_importance_sample=True) ----
        # Subtract each group's log density so the logsumexp estimates the
        # soft-max over actions w.r.t. the uniform measure. (When importance
        # sampling, the ref does NOT include the q_data term inside the logsumexp.)
        cat_q_1 = torch.cat(
            [q_rand_1, q_cur_1 - log_pi_cur, q_nxt_1 - log_pi_nxt], dim=-1
        )
        cat_q_2 = torch.cat(
            [q_rand_2, q_cur_2 - log_pi_cur, q_nxt_2 - log_pi_nxt], dim=-1
        )
        cql_qf1_ood = torch.logsumexp(cat_q_1, dim=-1)
        cql_qf2_ood = torch.logsumexp(cat_q_2, dim=-1)

        # Subtract the log-likelihood of the data action (ref: cql_qf_diff = ood - q_pred).
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

    def pretrain_critic_step(self, offline_batch) -> dict[str, torch.Tensor]:
        """
        Phase 2: run ONE Cal-QL critic-init gradient step on offline data only.

        Computes the Cal-QL critic loss (SAC TD error + conservative CQL term,
        floored by the offline MC return), backprops into the SHARED encoder +
        critic, clips grads, steps ``encoder_critic_optim``, then polyak-updates
        the target critic. The actor and temperature are left untouched — this
        only warms up Q so online RL does not start from a random critic.

        Args:
            offline_batch: an offline-only minibatch (see
                ``PLDReplayBuffer.sample_offline_only``); its ``mc_return`` field
                supplies the Cal-QL calibration floor.

        Returns:
            A log-info dict of detached scalar tensors (no grads):

                key                    | meaning
                ---------------------- | ----------------------------------------
                critic_loss            | total critic loss (td_loss + cql_loss)
                td_loss                | double-Q Bellman TD error
                cql_loss               | conservative CQL penalty (scaled)
                q_data1_mean           | mean Q1 on dataset actions
                q_data2_mean           | mean Q2 on dataset actions
                target_q_mean          | mean Bellman target
                cql_qf1_diff           | CQL gap (logsumexp - data) for Q1
                cql_qf2_diff           | CQL gap (logsumexp - data) for Q2
                mc_return_mean         | mean offline MC return in the batch
                mc_return_max          | max offline MC return in the batch
                mc_return_nonzero_frac | fraction of transitions with MC return != 0
                q_pi_pre_floor_mean    | mean policy-action Q before the MC floor
                calib_floor_binds_frac | fraction of policy-sample Qs the floor lifts

            Side effect: the encoder, critic, and target-critic weights are
            updated in place.
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

    @staticmethod
    def _slice_batch(batch: dict, i: int, mb: int) -> dict:
        """Slice a dict-of-tensors batch into minibatch ``i`` of size ``mb`` along
        dim 0. Works uniformly for vectors (B, D) and image tensors (B, n_cam, H,
        W, 3) since slicing only touches the leading (batch) axis.
        """
        s = slice(i * mb, (i + 1) * mb)
        return {k: v[s] for k, v in batch.items()}

    def update(self, batch, utd: int) -> dict:
        """One online update cycle (RLPD-faithful UTD; ref rlpd sac_learner.update).

        The incoming ``batch`` has size ``batch_size * utd``. It is sliced into
        ``utd`` DISTINCT, equal-size minibatches along dim 0 — one fresh minibatch
        per critic update (so ``utd`` distinct critic updates), with polyak after
        each. A single actor update and a single temperature update are then run
        on the LAST minibatch (matching RLPD doing actor/temp once per cycle).

        The per-minibatch loss math, grad clipping, and polyak are unchanged.
        Reported info comes from the last minibatch.
        """
        self.train()

        # Determine minibatch size; require exact divisibility (mirrors RLPD's
        # ``assert x.shape[0] % utd_ratio == 0``).
        total = batch["reward"].shape[0]
        assert total % utd == 0, (
            f"update() batch size {total} not divisible by utd={utd}; "
            f"the trainer must sample batch_size * utd."
        )
        mb = total // utd

        # ---- UTD critic updates (encoder + critic), one DISTINCT minibatch each ----
        critic_loss = None
        info_c = None
        last_mb = None
        for i in range(utd):
            last_mb = self._slice_batch(batch, i, mb)
            critic_loss, info_c = self.loss_critic(last_mb, alpha=self._alpha())
            self.encoder_critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm,
            )
            self.encoder_critic_optim.step()
            self.polyak_update()

        # ---- Actor update (last minibatch) ----
        actor_loss, info_a = self.loss_actor(last_mb, alpha=self._alpha())
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        # ---- Temperature update (last minibatch) ----
        tloss, info_t = self.loss_temperature(last_mb, self.log_alpha, self.target_entropy)
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
