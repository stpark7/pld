import os
import json

try:
    from collections.abc import Iterable
except ImportError:
    Iterable = (tuple, list)


def make_async(
    id,
    num_envs=1,
    asynchronous=True,
    wrappers=None,
    render=False,
    obs_dim=23,
    action_dim=7,
    env_type=None,
    max_episode_steps=None,
    # below for furniture only
    gpu_id=0,
    headless=True,
    record=False,
    normalization_path=None,
    furniture="one_leg",
    randomness="low",
    obs_steps=1,
    act_steps=8,
    sparse_reward=False,
    # below for robomimic only
    robomimic_env_cfg_path=None,
    use_image_obs=False,
    render_offscreen=False,
    reward_shaping=False,
    shape_meta=None,
    # below for pusht only
    success_threshold=0.7695,
    **kwargs,
):
    """Create a vectorized environment from multiple copies of an environment,
    from its id.

    NOTE (PLD port): only ``env_type == "robocasa"`` is supported in this
    standalone project. The other branches from dice-rl (pusht, furniture,
    robomimic, d3il, gym) depend on wrappers that were intentionally not ported.

    Parameters
    ----------
    id : str
        The environment ID. This must be a valid ID from the registry.

    num_envs : int
        Number of copies of the environment.

    asynchronous : bool
        If `True`, wraps the environments in an :class:`AsyncVectorEnv` (which uses
        `multiprocessing`_ to run the environments in parallel). If ``False``,
        wraps the environments in a :class:`SyncVectorEnv`.

    wrappers : dictionary, optional
        Each key is a wrapper class, and each value is a dictionary of arguments

    Returns
    -------
    :class:`gym.vector.VectorEnv`
        The vectorized environment.
    """

    if env_type == "robocasa":
        from gymnasium import spaces
        from env.gym_utils.async_vector_env import AsyncVectorEnv
        from env.gym_utils.sync_vector_env import SyncVectorEnv
        from env.gym_utils.wrapper import wrapper_dict

        def _make_robocasa_env():
            import gymnasium
            import robocasa  # triggers gymnasium env registration

            os.environ.setdefault("MUJOCO_GL", "egl")
            if render_offscreen or use_image_obs:
                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    cuda_device = os.environ["CUDA_VISIBLE_DEVICES"].split(',')[0]
                    os.environ["EGL_DEVICE_ID"] = cuda_device
                    os.environ["MUJOCO_EGL_DEVICE_ID"] = cuda_device

            robocasa_kwargs = dict(kwargs)
            robocasa_kwargs.pop("abs_action", None)
            env = gymnasium.make(
                id,
                split="pretrain",
                disable_env_checker=True,
                **robocasa_kwargs,
            )

            if wrappers is not None:
                for wrapper, args in wrappers.items():
                    env = wrapper_dict[wrapper](env, **args)
            return env

        def dummy_robocasa_env_fn():
            import gymnasium as gym
            import numpy as np
            from env.gym_utils.wrapper.multi_step import MultiStep
            from env.gym_utils.wrapper.multi_step_full import MultiStepFull

            env = gym.Env()
            observation_space = spaces.Dict()
            if shape_meta is not None:
                for key, value in shape_meta["obs"].items():
                    s = value["shape"]
                    if key.endswith("rgb"):
                        min_val, max_val = 0, 255
                        dtype = np.uint8
                    elif key.endswith("state"):
                        min_val, max_val = -1, 1
                        dtype = np.float32
                    else:
                        raise RuntimeError(f"Unsupported type {key}")
                    observation_space[key] = spaces.Box(
                        low=min_val, high=max_val, shape=s, dtype=dtype,
                    )
            else:
                observation_space["state"] = spaces.Box(
                    -1, 1, shape=(obs_dim,), dtype=np.float32,
                )
            env.observation_space = observation_space
            env.action_space = spaces.Box(-1, 1, shape=(action_dim,), dtype=np.float32)
            env.metadata = {
                "render.modes": ["human", "rgb_array"],
                "video.frames_per_second": 30,
            }
            if wrappers is not None and "multi_step" in wrappers:
                return MultiStep(env=env, n_obs_steps=wrappers.multi_step.n_obs_steps)
            elif wrappers is not None and "multi_step_full" in wrappers:
                return MultiStepFull(env=env, n_obs_steps=wrappers.multi_step_full.n_obs_steps)
            return env

        env_fns = [_make_robocasa_env for _ in range(num_envs)]
        return (
            AsyncVectorEnv(env_fns, dummy_env_fn=dummy_robocasa_env_fn)
            if asynchronous
            else SyncVectorEnv(env_fns)
        )

    raise NotImplementedError(
        f"env_type={env_type!r} is not supported in the PLD standalone project. "
        "Only env_type='robocasa' was ported; other sim backends from dice-rl "
        "(pusht, furniture, robomimic, d3il, gym) and their wrappers were "
        "intentionally omitted."
    )
