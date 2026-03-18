import numpy as np
from dataclasses import dataclass, field
from pyOpenHaptics.hd_callback import hd_callback
import pyOpenHaptics.hd as hd

@dataclass
class HapticState:
    position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rot_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    button: bool = False
    force: list = field(default_factory=lambda: [0.0, 0.0, 0.0])

haptic_state = HapticState()

@hd_callback
def haptic_callback():
    global haptic_state
    transform = hd.get_transform()
    velocity = hd.get_velocity()
    button = hd.get_buttons()
    
    haptic_state.position = [transform[3][0], transform[3][1], transform[3][2]]
    haptic_state.velocity = [velocity[0], velocity[1], velocity[2]]
    
    rot = np.array([
        [transform[0][0], transform[1][0], transform[2][0]],
        [transform[0][1], transform[1][1], transform[2][1]],
        [transform[0][2], transform[1][2], transform[2][2]]
    ])
    haptic_state.rot_matrix = rot
    haptic_state.button = True if button == 1 else False

    hd.set_force(haptic_state.force)
