import time

# Corrected imports matching your local package structure
from lerobot_robot_so101_mujoco.so101_mujoco_robot import So101MujocoRobot
from lerobot_robot_so101_mujoco.config_so101_mujoco_robot import So101MujocoRobotConfig

from lerobot_teleoperator_so101_haptic.so101_haptic import So101HapticTeleop
from lerobot_teleoperator_so101_haptic.config_so101_haptic_teleop import So101HapticTeleopConfig

def main():
    # 1. Initialize configurations
    robot_cfg = So101MujocoRobotConfig()
    teleop_cfg = So101HapticTeleopConfig()

    # 2. Instantiate the classes
    robot = So101MujocoRobot(robot_cfg)
    teleop = So101HapticTeleop(teleop_cfg)

    # 3. Connect both devices
    print("Starting connections...")
    robot.connect()
    teleop.connect()

    print("\n--- Bilateral Teleoperation Active ---")
    print("Press Ctrl+C to exit.\n")
    
    try:
        while True:
            # Step A: Get the physical movement from your Haptic Pen
            action = teleop.get_action()
            
            # Step B: Send that movement to the MuJoCo simulation
            robot.send_action(action)
            
            # Step C: Get the state of the simulation (including our new 'motor_force')
            obs = robot.get_observation()
            
            # Step D: Send the simulation force back to the Haptic Pen!
            teleop.send_feedback(obs)
            
            # Debug: See if the force is actually fluctuating!
            if "motor_force" in obs:
                #print(f"Force from MuJoCo: {obs['motor_force']}")
                pass
            
            # Run at roughly 100Hz
            time.sleep(0.01) 
            
    except KeyboardInterrupt:
        print("\nStopping teleoperation...")
    finally:
        robot.disconnect()
        teleop.disconnect()
        print("Safely disconnected.")

if __name__ == "__main__":
    main()