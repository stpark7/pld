"""
ResNet18-based multi-camera image encoder for PLD residual RL.

Each camera is encoded by a separate ImageNet-pretrained ResNet18, then features
are concatenated and projected to a fixed dim. BatchNorm can be frozen (eval mode)
to avoid corrupting ImageNet statistics under small RL batches.
"""

import torch
import torch.nn as nn
import torchvision

from model.common.modules import RandomShiftsAug


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class ThreeCamResNet18Encoder(nn.Module):
    def __init__(
        self,
        n_cam: int = 3,
        out_dim: int = 256,
        augment: bool = True,
        freeze_bn: bool = True,
        freeze_backbone: bool = False,
        pretrained: bool = True,
        shift_pad: int = 4,
    ):
        super().__init__()
        self.n_cam = n_cam
        self.out_dim = out_dim
        self.freeze_bn = freeze_bn
        self.freeze_backbone = freeze_backbone

        weights = "IMAGENET1K_V1" if pretrained else None
        self.encoders = nn.ModuleList(
            [self._build_resnet18(weights) for _ in range(n_cam)]
        )

        if freeze_backbone:
            for enc in self.encoders:
                for p in enc.parameters():
                    p.requires_grad = False

        feat_dim = 512  # ResNet18 final conv feature dim
        self.proj = nn.Sequential(
            nn.Linear(n_cam * feat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Tanh(),
        )

        self.aug = RandomShiftsAug(pad=shift_pad) if augment else None

        self.register_buffer(
            "_img_mean",
            torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_img_std",
            torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _build_resnet18(weights):
        net = torchvision.models.resnet18(weights=weights)
        net.fc = nn.Identity()
        return net

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, N_cam, H, W, 3) uint8 or float in [0, 255]
        Returns:
            feat: (B, out_dim)
        """
        if rgb.dtype != torch.float32:
            rgb = rgb.float()
        if rgb.max() > 1.5:
            rgb = rgb / 255.0

        B, N, H, W, C = rgb.shape
        assert N == self.n_cam, f"Expected {self.n_cam} cameras, got {N}"
        assert C == 3, f"Expected 3 channels, got {C}"

        # (B, N, H, W, 3) -> (B, N, 3, H, W)
        rgb = rgb.permute(0, 1, 4, 2, 3).contiguous()

        feats = []
        for i in range(self.n_cam):
            xi = rgb[:, i]  # (B, 3, H, W)
            if self.training and self.aug is not None:
                xi = self.aug(xi)
            xi = (xi - self._img_mean) / self._img_std
            fi = self.encoders[i](xi)  # (B, 512)
            feats.append(fi)

        feat = torch.cat(feats, dim=-1)  # (B, N*512)
        return self.proj(feat)  # (B, out_dim)
