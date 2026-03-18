import numpy as np
import threading
import time
from typing import Any
import scipy.spatial.transform as st  # Added for quaternion math

from lerobot.teleoperators.teleoperator import Teleoperator
from .config_so101_haptic_teleop import So101HapticTeleopConfig

import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description

from .haptics.get_position import haptic_state, haptic_callback
from .haptics.ik_feedback import calculate_ik_feedback
from .haptics.pyroki_snippets import solve_ik

from pyOpenHaptics.hd_device import HapticDevice

import viser
from viser.extras import ViserUrdf

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
        self._last_crash_print = 0.0
        self._last_lift_print = 0.0
        self._lift_active = False

        self.scale = 50.0 
        self.workspace_scale = 0.001 
        self.workspace_offset = np.array([0.0093, -0.2703, 0.15]) 
        
        self._filtered_torques = np.zeros(6)
        
        
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
        self._mujoco_robot_ref = None

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
            pitch_gain = 0.0
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

            # Update UI target
            if getattr(self, 'ik_web_target', None) is not None:
                self.ik_web_target.position = target_pos
                self.ik_web_target.wxyz = target_quat

            if q_sol is not None:
                gripper_val = 1.0 if button_pressed else 0.0
                
                # Update UI Ghost
                if getattr(self, 'urdf_vis', None) is not None:
                    q_vis = np.array(q_sol)
                    if "6" in self.urdf_joints:
                        gripper_idx = self.urdf_joints.index("6")
                        q_vis[gripper_idx] = gripper_val
                    self.urdf_vis.update_cfg(q_vis)

                with self._lock:
                    self._latest_target_pos = target_pos
                    self._latest_q_sol = q_sol
                    self._latest_gripper = gripper_val
                    
                # --- IK FEEDBACK USING ACTUAL ROBOT OBSERVATION ---
                # Retrieve the global So101MujocoRobot instance safely to 
                # bypass `send_feedback` and be standard-command compatible.
                if self._mujoco_robot_ref is None:
                    import gc
                    for obj in gc.get_objects():
                        if type(obj).__name__ == "So101MujocoRobot":
                            self._mujoco_robot_ref = obj
                            break
                            
                if self._mujoco_robot_ref is not None and hasattr(self._mujoco_robot_ref, "_latest_obs"):
                    obs = self._mujoco_robot_ref._latest_obs
                    actual_q = np.zeros(len(self.urdf_joints))
                    for i, j_name in enumerate(self.urdf_joints):
                        if j_name in self.ik_joint_mapping:
                            joint_key = f"{self.ik_joint_mapping[j_name]}.pos"
                            actual_q[i] = obs.get(joint_key, 0.0)
                        elif j_name == "6":
                            actual_q[i] = obs.get("gripper.pos", 0.0)
                            
                    force = calculate_ik_feedback(
                        robot=self.robot,
                        q_actual=actual_q,
                        expected_target_pos=target_pos,
                        target_link_name=self.config.target_link
                    )
                    haptic_state.force = force

            time.sleep(0.01)

    def connect(self) -> None:
        self.urdf = load_robot_description(self.config.urdf_name)
        self.robot = pk.Robot.from_urdf(self.urdf)
        self.urdf_joints = [j.name for j in self.urdf.actuated_joints]
        
        if getattr(self.config, "enable_viser", True):
            self.viser_server = viser.ViserServer(port=self.config.viser_port)
            self.viser_server.scene.add_grid("/ground", width=2, height=2)
            self.urdf_vis = ViserUrdf(self.viser_server, self.urdf, root_node_name="/ghost_robot")
            
            # Add visualizer target
            self.ik_web_target = self.viser_server.scene.add_transform_controls(
                "/ik_target", 
                scale=0.1, 
                position=(0.00931305, -0.27034248, 0.26730747), 
                wxyz=(0.707, -0.707, 0.0, 0.0)
            )
        else:
            self.viser_server = None
            self.ik_web_target = None
            self.urdf_vis = None
        
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
        if hasattr(self, 'viser_server') and self.viser_server:
            self.viser_server.stop()
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
            if "1" in self.urdf_joints:
                action_dict["shoulder_pan.pos"] = float(q_sol[self.urdf_joints.index("1")]) * self.scale
            if "2" in self.urdf_joints:
                action_dict["shoulder_lift.pos"] = float(q_sol[self.urdf_joints.index("2")]) * self.scale
            if "3" in self.urdf_joints:
                action_dict["elbow_flex.pos"] = float(q_sol[self.urdf_joints.index("3")]) * self.scale
            if "4" in self.urdf_joints:
                action_dict["wrist_flex.pos"] = float(q_sol[self.urdf_joints.index("4")]) * self.scale
            if "5" in self.urdf_joints:
                action_dict["wrist_roll.pos"] = float(q_sol[self.urdf_joints.index("5")]) * self.scale
            
        return action_dict
    
    @property
    def action_features(self) -> dict:
        # Register the features with the exact .pos suffix the pipeline expects
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
        pass