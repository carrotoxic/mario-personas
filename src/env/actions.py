"""Discrete 32-action space mapping button bitmasks to Java boolean arrays."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from jpype import JArray, JBoolean

from src.env.jvm import JavaClass


class ActionSpace:
    """Discrete actions 0..31; the id equals the low 5 bits of a ``.rep`` replay byte.
    Bit order is load-bearing (bit0=LEFT, bit1=RIGHT, bit2=DOWN, bit3=SPEED,
    bit4=JUMP): ``data_processing.replay`` maps human replay bytes onto it."""

    def __init__(self, mario_actions_class: JavaClass):
        n = int(mario_actions_class.numberOfActions())
        left = int(mario_actions_class.LEFT.getValue())
        right = int(mario_actions_class.RIGHT.getValue())
        down = int(mario_actions_class.DOWN.getValue())
        jump = int(mario_actions_class.JUMP.getValue())
        speed = int(mario_actions_class.SPEED.getValue())
        # Slot index in the Java boolean action array, per bit position.
        order = (left, right, down, speed, jump)

        def mask_to_array(mask: int) -> np.ndarray:
            array = np.zeros((n,), dtype=np.bool_)
            for bit in range(5):
                if (mask >> bit) & 1:
                    slot = order[bit]
                    if 0 <= slot < n:
                        array[slot] = True
            return array

        def mask_name(mask: int) -> str:
            parts = [("L", "R", "D", "S", "J")[i] for i in range(5) if (mask >> i) & 1]
            return "+".join(parts) if parts else "NOOP"

        self.action_set: list[tuple[np.ndarray, str]] = [
            (mask_to_array(mask), mask_name(mask)) for mask in range(32)
        ]
        self.n_actions = len(self.action_set)
        self.action_names = [name for _, name in self.action_set]
        # Pre-converted once to avoid per-step JArray allocation.
        self.java_actions = [JArray(JBoolean)(array.tolist()) for array, _ in self.action_set]

    def to_gym_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.n_actions)

    def get_java_action(self, action_idx: int) -> JavaClass:
        return self.java_actions[int(action_idx) % self.n_actions]

    def get_action_name(self, action_idx: int) -> str:
        return self.action_names[int(action_idx) % self.n_actions]
