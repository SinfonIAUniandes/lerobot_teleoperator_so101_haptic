import numpy as np
import threading
import time
from typing import Any
from dataclasses import dataclass, field

from lerobot.teleoperators.teleoperator import Teleoperator
from .config_so101_haptic_teleop import So101HapticTeleopConfig

import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description
from .pyroki_snippets import solve_ik

from pyOpenHaptics.hd_device import HapticDevice
import pyOpenHaptics.hd as hd
from pyOpenHaptics.hd_callback import hd_callback

# ---------------------------------------------------------------------------
# Haptic Device State & Callback
# ---------------------------------------------------------------------------
@dataclass
class HapticState:
    """Stores the latest state from the haptic device."""
    position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    button: bool = False

haptic_state = HapticState()

@hd_callback
def haptic_callback():
    """High-frequency callback executed by the HDAPI."""
    global haptic_state
    transform = hd.get_transform()
    button = hd.get_buttons()
    
    # Extract position from the 4x4 transform matrix
    haptic_state.position = [transform[3][0], transform[3][1], transform[3][2]]
    # Map the primary stylus button to True/False
    haptic_state.button = True if button == 1 else False

# ---------------------------------------------------------------------------
# Teleoperator Class
# ---------------------------------------------------------------------------
class So101HapticTeleop(Teleoperator):
    config_class = So101HapticTeleopConfig
    name = "so101_haptic"

    def __init__(self, config: So101HapticTeleopConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self.haptic_device = None
        
        # Threading mechanisms
        self._ik_thread = None
        self._lock = threading.Lock()
        
        # State variables protected by the lock
        self._latest_q_sol = None
        self._latest_gripper = 0.0

        self.scale = 50.0  # Scale factor for the robot arm joint angles
        
        # Workspace mapping (Haptic Device mm -> Robot Base meters)
        # You will likely need to tune these values based on your robot's origin
        self.workspace_scale = 0.001 # Convert mm to meters
        self.workspace_offset = np.array([0.0093, -0.2703, 0.2673]) # Default safe starting position
        
        # Fixed orientation for IK (matching your original UI's wxyz)
        self.target_quat = np.array([0.707, -0.707, 0.0, 0.0])

        self.ik_joint_mapping = {
            "1": "shoulder_pan", "2": "shoulder_lift", "3": "elbow_flex",
            "4": "wrist_flex", "5": "wrist_roll"
        }

    def _ik_worker(self):
        """Background thread that continuously solves IK based on the haptic position."""
        global haptic_state
        
        while self._is_connected:
            # Safely read the haptic state
            raw_pos = np.array(haptic_state.position)
            button_pressed = haptic_state.button
            
            # Note: The 3D Systems Touch coordinate frame is generally:
            # X: Right/Left, Y: Up/Down, Z: Towards/Away from user
            # You may need to swap axes here depending on your URDF base frame.
            # E.g., target_pos = np.array([raw_pos[0], raw_pos[2], raw_pos[1]]) * ...
            target_pos = (raw_pos * self.workspace_scale) + self.workspace_offset

            # Solve IK (this blocks the worker thread, but not the main loop)
            q_sol = solve_ik(
                robot=self.robot,
                target_link_name=self.config.target_link,
                target_position=target_pos,
                target_wxyz=self.target_quat,
            )

            if q_sol is not None:
                # Map the haptic button to the gripper (True = 1.0, False = 0.0)
                gripper_val = 1.0 if button_pressed else 0.0
                
                # Safely update the latest solution for get_action() to read
                with self._lock:
                    self._latest_q_sol = q_sol
                    self._latest_gripper = gripper_val

            # Prevent CPU pegging
            time.sleep(0.01)

    def connect(self) -> None:
        self.urdf = load_robot_description(self.config.urdf_name)
        self.robot = pk.Robot.from_urdf(self.urdf)
        self.urdf_joints = [j.name for j in self.urdf.actuated_joints]
        
        # --- Initialize Haptic Device ---
        print("\n--- Connecting to Haptic Device ---")
        self.haptic_device = HapticDevice(
            device_name="Default Device", 
            callback=haptic_callback, 
            scheduler_type="async"
        )
        time.sleep(0.2) # Give the HDAPI a moment to initialize
        
        # --- JAX Warm-up ---
        print("\n--- Compiling JAX IK Solver ---")
        print("This will take ~10-20 seconds. Please wait...")
        dummy_pos = np.array([0.3, 0.0, 0.2])
        dummy_quat = np.array([0.0, 0.0, 0.0, 0.0])
        solve_ik(
            robot=self.robot,
            target_link_name=self.config.target_link,
            target_position=dummy_pos,
            target_wxyz=dummy_quat,
        )
        print("--- JAX Compilation Complete! ---\n")

        self._is_connected = True

        # Start the background solver thread
        self._ik_thread = threading.Thread(target=self._ik_worker, daemon=True)
        self._ik_thread.start()

    def disconnect(self) -> None:
        self._is_connected = False
        if self._ik_thread:
            self._ik_thread.join(timeout=1.0)
            
        if self.haptic_device:
            print("\n--- Closing Haptic Device ---")
            self.haptic_device.close()

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def get_action(self) -> dict:
        """Returns joint commands in Radians, which the robot driver handles."""
        with self._lock:
            q_sol = self._latest_q_sol
            gripper_val = self._latest_gripper

        # We use 0.0 as the base (the robot's calibrated center)
        action_dict = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": float(gripper_val) * self.scale,
        }

        if q_sol is not None:
            # Map the IK solution URDF indices to the .pos string keys
            if "1" in self.urdf_joints: action_dict["shoulder_pan.pos"] = float(q_sol[self.urdf_joints.index("1")]) * self.scale
            if "2" in self.urdf_joints: action_dict["shoulder_lift.pos"] = float(q_sol[self.urdf_joints.index("2")]) * self.scale
            if "3" in self.urdf_joints: action_dict["elbow_flex.pos"] = float(q_sol[self.urdf_joints.index("3")])  * self.scale
            if "4" in self.urdf_joints: action_dict["wrist_flex.pos"] = float(q_sol[self.urdf_joints.index("4")]) * self.scale
            if "5" in self.urdf_joints: action_dict["wrist_roll.pos"] = float(q_sol[self.urdf_joints.index("5")]) * self.scale
            
        return action_dict
    
    @property
    def action_features(self) -> dict:
        return {
            "shoulder_pan.pos": float, "shoulder_lift.pos": float, "elbow_flex.pos": float,
            "wrist_flex.pos": float, "wrist_roll.pos": float, "gripper.pos": float,
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # In the future, you could read joint torques here and apply them to 
        # the haptic device using hd.set_force() to add force feedback!
        pass