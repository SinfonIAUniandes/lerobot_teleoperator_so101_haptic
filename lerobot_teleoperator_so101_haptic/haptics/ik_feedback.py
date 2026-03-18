import numpy as np
import jaxlie

def calculate_ik_feedback(robot, q_actual, expected_target_pos, target_link_name, kp=150.0, max_force=2.0):
    """
    Calculate the feedback force to be sent to the haptic device.
    
    Args:
        robot: pyroki Robot instance
        q_actual: numpy array of actual joint positions from the robot feedback
        expected_target_pos: numpy array (3,) of the target cartesian position we sent
        target_link_name: string representing the name of the end effector link
        kp: Proportional gain for the force feedback (N/m)
        max_force: Maximum force to apply to the haptic device
        
    Returns:
        list of 3 floats representing the force in the haptic device frame [Fx, Fy, Fz]
    """
    try:
        # Get actual position from forward kinematics
        target_idx = robot.links.names.index(target_link_name)
        
        # pyroki's forward_kinematics returns a stacked tensor of SE3 poses for each link
        se3_poses = robot.forward_kinematics(q_actual)
        actual_pos = jaxlie.SE3(se3_poses).translation()[target_idx]
        
        # Check mismatch
        # Force pushes the user's hand towards the "actual" position of the robot
        # meaning if the robot is blocked, the user feels pushed back.
        pos_error = actual_pos - expected_target_pos
        
        mismatch_norm = np.linalg.norm(pos_error)
        
        # Deadband factor (e.g., 100mm) to avoid jitter
        if mismatch_norm < 0.06:
            return [0.0, 0.0, 0.0]
            
        robot_force = pos_error * kp
        
        # Cap the force for safety
        force_norm = np.linalg.norm(robot_force)
        if force_norm > max_force:
            robot_force = (robot_force / force_norm) * max_force
            
        # -----------------------------------------------------
        # Coordinate map from Robot to Haptic Frame:
        # Based on the So101HapticTeleop class:
        # mapped_pos = [raw_pos[0], -raw_pos[2], raw_pos[1]]
        # Meaning:
        # Robot X = Haptic X
        # Robot Y = -Haptic Z
        # Robot Z = Haptic Y
        # 
        # So to map a force from Robot to Haptic, we invert:
        # Haptic X = Robot X
        # Haptic Y = Robot Z
        # Haptic Z = -Robot Y
        # -----------------------------------------------------
        haptic_force = [
            float(robot_force[0]),
            float(robot_force[2]),
            float(-robot_force[1])
        ]
        
        return haptic_force
        
    except Exception as e:
        print(f"Error calculating IK feedback: {e}")
        return [0.0, 0.0, 0.0]
