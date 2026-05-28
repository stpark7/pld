"""
Residual Gaussian actor for PLD Stage 1.

Outputs a_delta in [-xi, xi]^action_dim conditioned on (image_feat, proprio, a_base).
3-layer MLP with LayerNorm; tanh-squash + log-prob correction (SAC-style).
"""

import math
import torch
import torch.nn as nn


class ResidualGaussianActor(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        proprio_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        xi: float = 0.5,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.xi = xi
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        in_dim = feat_dim + proprio_dim + action_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.logstd_head = nn.Linear(hidden_dim, action_dim)

        # log(xi) shift for log-prob (constant per-dim; doesn't affect gradients but
        # keeps the log-prob mathematically correct for the scaled action).
        self._log_xi = math.log(max(xi, 1e-8))

    def forward(
        self,
        feat: torch.Tensor,
        proprio: torch.Tensor,
        a_base: torch.Tensor,
        deterministic: bool = False,
        reparameterize: bool = True,
        get_logprob: bool = False,
    ):
        x = torch.cat([feat, proprio, a_base], dim=-1)
        h = self.trunk(x)
        mean = self.mean_head(h)
        log_std = self.logstd_head(h).clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()

        if deterministic:
            u = mean
            a_unscaled = torch.tanh(u)
            a_delta = self.xi * a_unscaled
            if get_logprob:
                # Deterministic: log-prob undefined; return zeros for shape.
                return a_delta, torch.zeros(a_delta.shape[0], device=a_delta.device)
            return a_delta

        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample() if reparameterize else dist.sample()
        a_unscaled = torch.tanh(u)
        a_delta = self.xi * a_unscaled

        if get_logprob:
            # log p(u)
            log_p = dist.log_prob(u).sum(-1)
            # tanh squash correction: d a_unscaled / d u = 1 - tanh(u)^2
            log_p = log_p - torch.log(1 - a_unscaled.pow(2) + 1e-6).sum(-1)
            # constant shift for scaling by xi (per action dim)
            log_p = log_p - self.action_dim * self._log_xi
            return a_delta, log_p
        return a_delta
