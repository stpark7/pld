"""
Single shared robocasa env pool (GPU/RAM safety).

The 4-env hard cap is critical: RoboCasa is heavy and running more than 4
concurrent robocasa worker processes freezes the machine. The original dice-rl
code created a *separate* eval_venv (n_envs more procs) alongside the training
venv, so a naive n_envs=4 + eval_n_envs=4 run would launch 8 procs.

This pool removes that footgun. There is exactly ONE vectorized env, owned by
the orchestrator, reused for BOTH training and evaluation. ``evaluate()`` resets
this same pool, runs eval episodes on it, then training resumes via a fresh
bootstrap (see ``TrainResidualRL``). Peak load is always n_envs robocasa procs
+ 1 GR00T model.

A module-level guard asserts the cap is never exceeded (defense-in-depth: even if
a future caller tries to build a second pool, the assert fires).
"""

import logging

from env.gym_utils import make_async

log = logging.getLogger(__name__)

# Hard cap on concurrent robocasa worker processes across the whole process.
MAX_ROBOCASA_ENVS = 4

# Module-level accounting of how many robocasa envs are currently alive.
_ALIVE_ENVS = 0


class EnvPool:
    """A single shared robocasa vectorized environment.

    Wraps ``make_async`` and tracks live env count to enforce the 4-env cap.
    """

    def __init__(self, cfg, seed: int):
        global _ALIVE_ENVS

        env_cfg = cfg.env
        self.n_envs = int(env_cfg.n_envs)
        self.env_name = env_cfg.name
        self.env_type = env_cfg.get("env_type", "robocasa")
        self.max_episode_steps = int(env_cfg.max_episode_steps)
        self.seed = seed

        assert self.n_envs <= MAX_ROBOCASA_ENVS, (
            f"env.n_envs={self.n_envs} exceeds the hard cap of "
            f"{MAX_ROBOCASA_ENVS} concurrent robocasa envs (machine freezes "
            "above this)."
        )
        assert _ALIVE_ENVS + self.n_envs <= MAX_ROBOCASA_ENVS, (
            f"Refusing to create {self.n_envs} more robocasa envs: "
            f"{_ALIVE_ENVS} already alive, cap is {MAX_ROBOCASA_ENVS}. "
            "There must be exactly ONE shared pool (no separate eval pool)."
        )

        log.info(
            f"Creating shared EnvPool: {self.n_envs} robocasa envs "
            f"({self.env_name}); cap={MAX_ROBOCASA_ENVS}"
        )
        self.venv = make_async(
            env_cfg.name,
            env_type=self.env_type,
            num_envs=self.n_envs,
            asynchronous=True,
            max_episode_steps=self.max_episode_steps,
            wrappers=env_cfg.get("wrappers", None),
            shape_meta=cfg.get("shape_meta", None),
            use_image_obs=env_cfg.get("use_image_obs", False),
            render=env_cfg.get("render", False),
            render_offscreen=env_cfg.get("save_video", False),
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
        )
        _ALIVE_ENVS += self.n_envs
        self._closed = False

        # Seed once at construction (parallel envs need distinct initial states).
        self.venv.seed([self.seed + i for i in range(self.n_envs)])

    # ---- pass-through API used by the orchestrator ----

    def reset(self):
        return self.venv.reset()

    def step(self, actions):
        return self.venv.step(actions)

    def seed(self, seeds):
        return self.venv.seed(seeds)

    def close(self):
        global _ALIVE_ENVS
        if self._closed:
            return
        self.venv.close()
        _ALIVE_ENVS -= self.n_envs
        self._closed = True
        log.info(f"Closed EnvPool ({self.n_envs} envs); alive now {_ALIVE_ENVS}")
