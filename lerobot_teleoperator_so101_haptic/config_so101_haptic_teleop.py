from dataclasses import dataclass
from lerobot.teleoperators.config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("so101_haptic")
@dataclass
class So101HapticTeleopConfig(TeleoperatorConfig):
    urdf_name: str = "so_arm101_description"
    target_link: str = "gripper"
    viser_port: int = 8080
    use_haptics: bool = True
    haptic_mode: str = "ik"  
    enable_viser: bool = True