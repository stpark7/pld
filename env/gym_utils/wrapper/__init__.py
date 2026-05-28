"""Wrapper registry for the PLD robocasa residual-RL project.

Only the wrappers needed by the robocasa branch of ``env.gym_utils.make_async``
are registered here. The unrelated sim wrappers from dice-rl (robomimic, d3il,
mujoco_locomotion, pusht, furniture) are intentionally NOT ported — they pull in
heavy/unrelated dependencies and are never referenced when creating robocasa envs.
"""

from .multi_step import MultiStep
from .multi_step_full import MultiStepFull
from .robocasa_image import RobocasaImageWrapper


wrapper_dict = {
    "multi_step": MultiStep,
    "multi_step_full": MultiStepFull,
    "robocasa_image": RobocasaImageWrapper,
}
