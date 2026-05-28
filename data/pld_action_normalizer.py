"""
PLD action normalization based on lerobot dataset stats.

Loads `meta/stats.json` from a lerobot dataset (e.g. the robocasa atomic-task
demonstrations used to train GR00T) and provides q01/q99-based normalisation
between RoboCasa's 12-D raw action space and dice-rl's 7-D normalised arm space
([ee_pos(3), ee_rot(3), gripper(1)] in [-1, 1]).
"""

import json
from pathlib import Path

import numpy as np


class PLDActionNormalizer:
    # Indices of the arm-only sub-action within the 12-D RoboCasa action vector,
    # per the lerobot modality.json:
    #   action[5:8]   -> end_effector_position
    #   action[8:11]  -> end_effector_rotation
    #   action[11:12] -> gripper_close
    ARM_SLICE = slice(5, 12)

    def __init__(self, lerobot_meta_dir: str, use_quantile: bool = True):
        stats_path = Path(lerobot_meta_dir) / "stats.json"
        with stats_path.open() as f:
            stats = json.load(f)

        if use_quantile and "q01" in stats["action"]:
            lo = np.asarray(stats["action"]["q01"], dtype=np.float32)
            hi = np.asarray(stats["action"]["q99"], dtype=np.float32)
        else:
            lo = np.asarray(stats["action"]["min"], dtype=np.float32)
            hi = np.asarray(stats["action"]["max"], dtype=np.float32)

        self.arm_lo = lo[self.ARM_SLICE].copy()
        self.arm_hi = hi[self.ARM_SLICE].copy()
        range_ = self.arm_hi - self.arm_lo
        self.zero_range = np.abs(range_) < 1e-6
        range_ = np.where(self.zero_range, 1.0, range_).astype(np.float32)
        self.arm_range = range_

    # ---- forward / inverse ----

    def normalize(self, raw_7d: np.ndarray) -> np.ndarray:
        """raw arm action -> [-1, 1]^7"""
        raw_7d = np.asarray(raw_7d, dtype=np.float32)
        norm = 2.0 * (raw_7d - self.arm_lo) / self.arm_range - 1.0
        norm = np.where(self.zero_range, 0.0, norm)
        return np.clip(norm, -1.0, 1.0).astype(np.float32)

    def unnormalize(self, norm_7d: np.ndarray) -> np.ndarray:
        """[-1, 1]^7 -> raw arm action (no clipping; caller decides)"""
        norm_7d = np.asarray(norm_7d, dtype=np.float32)
        unscaled = (norm_7d + 1.0) / 2.0
        raw = unscaled * self.arm_range + self.arm_lo
        raw = np.where(self.zero_range, self.arm_lo, raw)
        return raw.astype(np.float32)

    # ---- GR00T dict <-> 7D arm ----

    @staticmethod
    def extract_arm_from_gr00t(action_dict) -> np.ndarray:
        """Return (T, 7) raw arm action from a GR00T action dict.

        Keys: action.end_effector_position (T,3), end_effector_rotation (T,3),
              gripper_close (T,1) or (T,).
        """
        ee_pos = np.asarray(action_dict["action.end_effector_position"], dtype=np.float32)
        ee_rot = np.asarray(action_dict["action.end_effector_rotation"], dtype=np.float32)
        gripper = np.asarray(action_dict["action.gripper_close"], dtype=np.float32)
        if gripper.ndim == 1:
            gripper = gripper[:, None]
        if ee_pos.ndim == 1:
            ee_pos = ee_pos[None]
            ee_rot = ee_rot[None]
            gripper = gripper[None]
        return np.concatenate([ee_pos, ee_rot, gripper], axis=-1)
