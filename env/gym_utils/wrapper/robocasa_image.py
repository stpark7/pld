"""
Environment wrapper for RoboCasa environments with image observations.

Built on gymnasium (both the underlying RoboCasa env and the vector envs in env/gym_utils/).
Extracts 9D arm-only state and concatenates 3 camera images for the residual policy,
and at the same time exposes a GR00T N1.5 Panda-Omron-ready observation dict (time axis
added, language included) via ``info["gr00t_raw"]`` for the frozen base policy.
Converts 7D normalized actions to 12D Dict actions for RoboCasa.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import imageio


# RoboCasa observation keys for arm-only state (9D)
STATE_KEYS = [
    "state.end_effector_position_relative",  # 3D
    "state.end_effector_rotation_relative",  # 4D
    "state.gripper_qpos",                    # 2D
]

# RoboCasa camera keys
CAMERA_KEYS = [
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
]

# GR00T N1.5 Panda-Omron modality keys
GR00T_VIDEO_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
GR00T_STATE_KEYS = (
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
    "state.base_position",
    "state.base_rotation",
)
GR00T_LANGUAGE_KEY = "annotation.human.task_description"


class RobocasaImageWrapper(gym.Env):
    def __init__(
        self,
        env,
        shape_meta: dict,
        normalization_path=None,
        state_keys=None,
        camera_keys=None,
        clamp_obs=False,
        render_hw=(256, 256),
        render_camera_name="robot0_agentview_left",
        success_steps_before_termination=5,
        keep_cams_separate=False,
        task_description=None,
    ):
        self.env = env  # gymnasium env
        self.clamp_obs = clamp_obs
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.success_steps_before_termination = success_steps_before_termination
        self.keep_cams_separate = keep_cams_separate
        self.task_description = task_description
        self._last_raw_obs = None

        self.state_keys = state_keys or STATE_KEYS
        self.camera_keys = camera_keys or CAMERA_KEYS

        # Tracking for success-based termination
        self.success_count = 0
        self.episode_reward = 0.0
        self.step_count = 0
        self.ever_succeeded = False

        # Normalization (for eval rollouts — dataset is already normalized)
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        # Action space: 7D normalized [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # Observation space from shape_meta
        observation_space = spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            if key.endswith("rgb"):
                min_value, max_value = 0, 255
                dtype = np.uint8
            elif key.endswith("state"):
                min_value, max_value = -1, 1
                dtype = np.float32
            else:
                raise RuntimeError(f"Unsupported obs key: {key}")
            observation_space[key] = spaces.Box(
                low=min_value, high=max_value, shape=shape, dtype=dtype,
            )
        self.observation_space = observation_space

    def normalize_obs(self, obs):
        obs = 2 * (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 1  # -> [-1, 1]
        # Always clip — necessary for quantile normalization where
        # values beyond q01/q99 would exceed [-1, 1]
        obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        action_range = self.action_max - self.action_min
        # When range is zero (e.g. gripper always -1), use action_min directly
        zero_range = np.abs(action_range) < 1e-6
        action_range = np.where(zero_range, 1.0, action_range)
        result = action * action_range + self.action_min
        result = np.where(zero_range, self.action_min, result)
        return result

    def _action_to_dict(self, action_7d):
        """Convert 7D flat action to RoboCasa Dict action.

        7D = ee_pos(3) + ee_rot(3) + gripper(1)
        Dict = base_motion(4) + control_mode(1) + ee_pos(3) + ee_rot(3) + gripper(1)
        """
        return {
            "action.base_motion": np.zeros(4, dtype=np.float32),
            "action.control_mode": np.zeros(1, dtype=np.float32),
            "action.end_effector_position": action_7d[0:3].astype(np.float32),
            "action.end_effector_rotation": action_7d[3:6].astype(np.float32),
            "action.gripper_close": action_7d[6:7].astype(np.float32),
        }

    def get_observation(self, raw_obs):
        """Extract 9D state and concatenated RGB from RoboCasa obs dict."""
        # State: concat arm-only keys
        state_parts = []
        for key in self.state_keys:
            state_parts.append(raw_obs[key])
        state = np.concatenate(state_parts, axis=0).astype(np.float32)

        if self.normalize:
            state = self.normalize_obs(state)

        # RGB: either concat along channel axis (default) or stack per camera
        rgb_parts = []
        for key in self.camera_keys:
            rgb_parts.append(raw_obs[key])  # (H, W, 3) uint8
        if self.keep_cams_separate:
            rgb = np.stack(rgb_parts, axis=0).astype(np.uint8)  # (N_cam, H, W, 3)
        else:
            rgb = np.concatenate(rgb_parts, axis=-1).astype(np.uint8)  # (H, W, N_cam*3)

        return {"state": state, "rgb": rgb}

    def _build_gr00t_obs(self, raw_obs):
        """Build a GR00T N1.5 Panda-Omron-ready observation dict from RoboCasa raw obs.

        - Selects only the modality keys GR00T uses (3 cameras + 5 state keys).
        - Adds an explicit time axis ``T=1`` to every video/state field.
        - Adds ``annotation.human.task_description`` from raw obs if present,
          otherwise from ``self.task_description``.

        Returns None if a language source is unavailable (neither raw obs nor
        ``self.task_description`` provides one). Non-GR00T callers ignore this.
        """
        if GR00T_LANGUAGE_KEY in raw_obs:
            language = raw_obs[GR00T_LANGUAGE_KEY]
        elif self.task_description is not None:
            language = self.task_description
        else:
            return None

        out = {}
        for key in GR00T_VIDEO_KEYS:
            if key not in raw_obs:
                raise KeyError(f"Raw obs missing GR00T video key: {key}")
            arr = np.asarray(raw_obs[key])
            if arr.ndim == 3:  # (H, W, C) -> (T=1, H, W, C)
                arr = arr[None]
            elif arr.ndim != 4:
                raise ValueError(f"Unexpected shape for {key}: {arr.shape}")
            out[key] = arr
        for key in GR00T_STATE_KEYS:
            if key not in raw_obs:
                raise KeyError(f"Raw obs missing GR00T state key: {key}")
            arr = np.asarray(raw_obs[key])
            if arr.ndim == 1:  # (D,) -> (T=1, D)
                arr = arr[None]
            elif arr.ndim != 2:
                raise ValueError(f"Unexpected shape for {key}: {arr.shape}")
            out[key] = arr
        lang_arr = np.asarray(language, dtype=object)
        if lang_arr.ndim == 0:
            lang_arr = lang_arr.reshape(1)
        out[GR00T_LANGUAGE_KEY] = lang_arr
        return out

    def _combined_camera_frame(self, raw_obs):
        frames = []
        for key in self.camera_keys:
            if key in raw_obs:
                frames.append(raw_obs[key])
        if not frames:
            return None
        return np.concatenate(frames, axis=1).astype(np.uint8)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, options=None, **kwargs):
        options = {} if options is None else dict(options)

        # Close video if exists
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        # Start video if specified
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        reset_kwargs = {}
        reset_seed = kwargs.get("seed", options.pop("seed", None))
        if reset_seed is not None:
            reset_kwargs["seed"] = reset_seed
        env_options = {k: v for k, v in options.items() if k != "video_path"}
        if env_options:
            reset_kwargs["options"] = env_options

        # gymnasium reset -> (obs, info)
        raw_obs, info = self.env.reset(**reset_kwargs)
        self._last_raw_obs = raw_obs

        # Reset tracking
        self.success_count = 0
        self.episode_reward = 0.0
        self.step_count = 0
        self.ever_succeeded = False

        return self.get_observation(raw_obs)

    def step(self, action):
        # Denormalize 7D action
        if self.normalize:
            action = self.unnormalize_action(action)

        # Convert to Dict action for RoboCasa
        dict_action = self._action_to_dict(action)

        # gymnasium step -> 5-tuple
        raw_obs, reward, terminated, truncated, info = self.env.step(dict_action)
        self._last_raw_obs = raw_obs
        obs = self.get_observation(raw_obs)

        # Expose a GR00T N1.5-ready obs (time axis added, language included)
        # so the frozen base policy can be queried directly from `info["gr00t_raw"]`.
        # Skipped when no language source exists — non-GR00T configs don't need this.
        gr00t_obs = self._build_gr00t_obs(raw_obs)
        if gr00t_obs is not None:
            info["gr00t_raw"] = gr00t_obs

        # Track success for termination
        self.step_count += 1
        self.episode_reward += reward

        if reward > 0:
            self.success_count += 1
            self.ever_succeeded = True
            if self.success_count >= self.success_steps_before_termination:
                terminated = True  # treat success-based termination as terminated, not truncated
                self.success_count = 0
                self.episode_reward = 0.0
                self.step_count = 0
                self.ever_succeeded = False
        else:
            if not self.ever_succeeded:
                self.success_count = 0

        if truncated:
            info["TimeLimit.truncated"] = True

        # Render to video
        if self.video_writer is not None:
            video_img = self._combined_camera_frame(raw_obs)
            if video_img is None:
                video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        # Auto-reset on episode end. dice-rl SyncVectorEnv does NOT auto-reset on
        # done, so if we don't reset here the next env.step() will run on a
        # terminal state and may never produce done again (observed: eval stalls
        # after some step). We replace obs with the post-reset observation; the
        # `done` flag (terminated/truncated) is preserved so Bellman masking is
        # correct for buffer transitions.
        done = bool(terminated) or bool(truncated)
        if done:
            new_obs = self.reset()  # also refreshes self._last_raw_obs
            obs = new_obs
            new_gr00t_obs = self._build_gr00t_obs(self._last_raw_obs)
            if new_gr00t_obs is not None:
                info["gr00t_raw"] = new_gr00t_obs

        # Return gymnasium 5-tuple to match dice-rl's SyncVectorEnv expectation
        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        try:
            raw_obs, _ = self.env.unwrapped.get_obs()
        except Exception:
            raw_obs = None

        if hasattr(raw_obs, "keys"):
            video_img = self._combined_camera_frame(raw_obs)
            if video_img is not None:
                return video_img

        try:
            return self.env.render()
        except Exception:
            return np.zeros((*self.render_hw, 3), dtype=np.uint8)

    def close(self):
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        self.env.close()
