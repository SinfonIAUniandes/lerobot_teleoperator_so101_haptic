import numpy as np
import threading
import time
from typing import Any
from dataclasses import dataclass, field
import scipy.spatial.transform as st  # Added for quaternion math

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
    velocity: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rot_matrix: np.ndarray = field(default_factory=lambda: np.eye(3)) # Added rotation
    button: bool = False
    force: list = field(default_factory=lambda: [0.0, 0.0, 0.0])

haptic_state = HapticState()

@hd_callback
def haptic_callback():
    """High-frequency callback executed by the HDAPI."""
    global haptic_state
    transform = hd.get_transform()
    velocity = hd.get_velocity()
    button = hd.get_buttons()
    
    # Extract translation (Position)
    haptic_state.position = [transform[3][0], transform[3][1], transform[3][2]]
    haptic_state.velocity = [velocity[0], velocity[1], velocity[2]]
    
    # Extract 3x3 Rotation Matrix from the 4x4 transform (column-major)
    rot = np.array([
        [transform[0][0], transform[1][0], transform[2][0]],
        [transform[0][1], transform[1][1], transform[2][1]],
        [transform[0][2], transform[1][2], transform[2][2]]
    ])
    haptic_state.rot_matrix = rot

    # Map the primary stylus button to True/False
    haptic_state.button = True if button == 1 else False

    hd.set_force(haptic_state.force)  # Apply any forces set by the main thread
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
        
        self._ik_thread = None
        self._lock = threading.Lock()
        
        self._latest_q_sol = None
        self._latest_gripper = 0.0

        self.scale = 50.0 
        self.workspace_scale = 0.001 
        self.workspace_offset = np.array([0.0093, -0.2703, 0.15]) 
        
        self._filtered_force = np.array([0.0, 0.0, 0.0])
        
        
        # Fixed orientation for IK (matching your original UI's wxyz)
        self.target_quat = np.array([0.707, -0.707, 0.0, 0.0])

        # The initial orientation offset for your gripper 
        # (Equivalent to your original wxyz: 0.707, -0.707, 0.0, 0.0)
        # Scipy uses xyzw format, so this is [-0.707, 0.0, 0.0, 0.707]
        self.gripper_offset_rot = st.Rotation.from_quat([-0.707, 0.0, 0.0, 0.707])

        self.ik_joint_mapping = {
            "1": "shoulder_pan", "2": "shoulder_lift", "3": "elbow_flex",
            "4": "wrist_flex", "5": "wrist_roll"
        }

    def _ik_worker(self):
        """Background thread that continuously solves IK based on the haptic position."""
        global haptic_state
        
        # Matrix to align the Haptic Frame to the Robot Frame
        # Robot X = Haptic X (raw_pos[0])
        # Robot Y = -Haptic Z (-raw_pos[2])
        # Robot Z = Haptic Y (raw_pos[1])
        R_align = np.array([
            [ 1,  0,  0],
            [ 0,  0, -1],
            [ 0,  1,  0]
        ])

        while self._is_connected:
            # Safely read the haptic state
            raw_pos = np.array(haptic_state.position)
            raw_rot = haptic_state.rot_matrix
            button_pressed = haptic_state.button
            
            # --- 1. Map Translation ---
            mapped_pos = np.array([raw_pos[0], -raw_pos[2], raw_pos[1]])
            target_pos = (mapped_pos * self.workspace_scale) + self.workspace_offset

            # --- 2. Map Rotation ---
            # Align the haptic rotation matrix to the robot's base frame
            mapped_rot_matrix = R_align @ raw_rot
            mapped_rot = st.Rotation.from_matrix(mapped_rot_matrix)
            
            # --- APPLY ROTATION GAIN ---
            # Convert to Euler angles (Extrinsic XYZ). 
            # Depending on your R_align, X is likely Roll, Y is Pitch, Z is Yaw.
            euler_angles = mapped_rot.as_euler('xyz', degrees=False)
            
            # Set your multipliers here! 
            # 1.0 is 1:1 mapping. 2.0 means moving the stylus 10 degrees moves the robot 20 degrees.
            roll_gain = 1.8 
            pitch_gain = 1.0
            yaw_gain = 1.0 
            
            euler_angles[0] *= roll_gain
            euler_angles[1] *= pitch_gain
            euler_angles[2] *= yaw_gain
            
            # Reconstruct the scaled rotation
            scaled_mapped_rot = st.Rotation.from_euler('xyz', euler_angles, degrees=False)
            
            # Combine scaled mapped rotation with your gripper's starting offset
            final_rot = self.gripper_offset_rot * scaled_mapped_rot
            
            # Scipy returns xyzw, but your IK solver expects wxyz
            quat_xyzw = final_rot.as_quat()
            target_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

            # Solve IK
            q_sol = solve_ik(
                robot=self.robot,
                target_link_name=self.config.target_link,
                target_position=target_pos,
                target_wxyz=target_quat,
            )

            if q_sol is not None:
                gripper_val = 1.0 if button_pressed else 0.0
                with self._lock:
                    self._latest_q_sol = q_sol
                    self._latest_gripper = gripper_val

            time.sleep(0.01)

    def connect(self) -> None:
        self.urdf = load_robot_description(self.config.urdf_name)
        self.robot = pk.Robot.from_urdf(self.urdf)
        self.urdf_joints = [j.name for j in self.urdf.actuated_joints]
        
        print("\n--- Connecting to Haptic Device ---")
        self.haptic_device = HapticDevice(
            device_name="Default Device", 
            callback=haptic_callback, 
            scheduler_type="async"
        )
        time.sleep(0.2) 
        
        print("\n--- Compiling JAX IK Solver ---")
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
        with self._lock:
            q_sol = self._latest_q_sol
            gripper_val = self._latest_gripper

        action_dict = {
            "shoulder_pan.pos": 0.0, "shoulder_lift.pos": 0.0, "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0, "wrist_roll.pos": 0.0, "gripper.pos": float(gripper_val) * self.scale,
        }

        if q_sol is not None:
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
        return {"motor_force": (3,)}

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:# Check the config parameter
        global haptic_state
        if not self.config.use_haptics:
            # Send zero force to ensure motors are 'off'
            haptic_state.force = [0.0, 0.0, 0.0]
            return
        
        
        if "motor_force" not in feedback:
            return
            
        raw_force = np.array(feedback["motor_force"])
        
        # --- 1. LOW-PASS FILTER (Smooths out violent spikes) ---
        alpha = 0.15 # 0.0 is entirely smooth/unresponsive, 1.0 is raw noise. 
        self._filtered_force = (alpha * raw_force) + ((1.0 - alpha) * self._filtered_force)
        
        # --- 2. SCALE ---
        force_scale = 0.03 # Keep it low while testing
        scaled_force = self._filtered_force * force_scale
        
        # Map Robot frame to Haptic frame
        # Robot X = Haptic X | Robot Y = -Haptic Z | Robot Z = Haptic Y
        mapped_force = np.array([
            float(scaled_force[0]),
            float(scaled_force[2]),
            float(-scaled_force[1])
        ])

        # --- 3. VIRTUAL DAMPING (Kills the shaking) ---
        haptic_vel = np.array(haptic_state.velocity)
        
        # Damping coefficient. It pushes opposite to your movement.
        damping_b = 0.002 
        damping_force = -damping_b * haptic_vel
        
        # Combine the forces!
        final_force = mapped_force + damping_force
        
        # --- 4. CLAMP FOR SAFETY ---
        max_haptic_force = 3.3 
        force_mag = np.linalg.norm(final_force)
        if force_mag > max_haptic_force:
            final_force = (final_force / force_mag) * max_haptic_force
            
        haptic_state.force = final_force.tolist()