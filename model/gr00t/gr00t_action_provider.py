"""Utilities for running GR00T policies inside DICE-RL evaluation code.

This module intentionally keeps GR00T imports lazy. The robocasa benchmark
stack has heavy optional dependencies, so importing DICE-RL should not require
GR00T to be installed unless the GR00T evaluation path is used.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt


def prepend_repo_path(repo_path: str | None) -> None:
    """Make an editable GR00T repository importable without installing it.

    Args:
        repo_path: Path to an Isaac-GR00T checkout. If ``None``, no-op.
    """
    if repo_path is None:
        return
    resolved = str(Path(repo_path).expanduser().resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


class GR00TActionProvider:
    """Thin wrapper around GR00T policy APIs used by DICE-RL evaluation."""

    def __init__(
        self,
        model_path: str,
        data_config: str = "panda_omron",
        embodiment_tag: str = "new_embodiment",
        denoising_steps: int = 4,
        device: str = "cuda",
        gr00t_repo_path: str | None = None,
    ):
        prepend_repo_path(gr00t_repo_path)
        self.model_path = model_path
        self.data_config_name = data_config
        self.embodiment_tag = embodiment_tag
        self.denoising_steps = denoising_steps
        self.device = device

        self.policy = self._build_policy()

    def get_action(self, observations: Mapping[str, Any]) -> dict[str, npt.NDArray[np.generic]]:
        """Return an action dictionary for a GR00T-compatible observation."""
        action = self.policy.get_action(dict(observations))
        if isinstance(action, tuple):
            action = action[0]
        if isinstance(action, Mapping) and "actions" in action and isinstance(action["actions"], Mapping):
            action = action["actions"]
        return {key: np.asarray(value) for key, value in action.items()}

    def get_modality_config(self) -> Mapping[str, Any]:
        return self.policy.get_modality_config()

    @property
    def video_delta_indices(self) -> npt.NDArray[np.generic]:
        if hasattr(self.policy, "video_delta_indices"):
            return np.asarray(self.policy.video_delta_indices)
        modality_config = self.get_modality_config()
        return np.asarray(modality_config["video"].delta_indices)

    @property
    def state_delta_indices(self) -> npt.NDArray[np.generic] | None:
        if hasattr(self.policy, "state_delta_indices"):
            value = getattr(self.policy, "state_delta_indices")
            return None if value is None else np.asarray(value)
        modality_config = self.get_modality_config()
        if "state" not in modality_config:
            return None
        return np.asarray(modality_config["state"].delta_indices)

    def _build_policy(self) -> Any:
        """Instantiate the robocasa-benchmark GR00T N1.5 policy."""
        data_config_module = importlib.import_module("gr00t.experiment.data_config")
        policy_module = self._import_policy_module()

        data_config_map = getattr(data_config_module, "DATA_CONFIG_MAP")
        if self.data_config_name not in data_config_map:
            available = ", ".join(sorted(data_config_map.keys()))
            raise KeyError(
                f"Unknown GR00T data_config '{self.data_config_name}'. "
                f"Available configs: {available}"
            )

        data_config = data_config_map[self.data_config_name]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
        gr00t_policy = getattr(policy_module, "Gr00tPolicy")

        signature = inspect.signature(gr00t_policy)
        kwargs: dict[str, object] = {
            "model_path": self.model_path,
            "embodiment_tag": self.embodiment_tag,
            "device": self.device,
        }
        if "modality_config" in signature.parameters:
            kwargs["modality_config"] = modality_config
        if "modality_transform" in signature.parameters:
            kwargs["modality_transform"] = modality_transform
        if "denoising_steps" in signature.parameters:
            kwargs["denoising_steps"] = self.denoising_steps

        return gr00t_policy(**kwargs)

    @staticmethod
    def _import_policy_module() -> Any:
        """Import the GR00T policy module across N1.x package layouts."""
        try:
            return importlib.import_module("gr00t.model.policy")
        except ModuleNotFoundError:
            return importlib.import_module("gr00t.policy.gr00t_policy")
