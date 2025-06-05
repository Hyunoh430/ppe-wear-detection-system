"""
PPE Waste Disposal System Package

A comprehensive system for detecting proper Personal Protective Equipment (PPE) 
usage and controlling waste disposal access accordingly.

Components:
- PPE Detection using YOLOv8 TFLite model
- Servo motor control for door mechanism
- Integrated system with safety features
- Comprehensive logging and monitoring

Author: PPE Safety Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "PPE Safety Team"

from ppe_detector import PPEDetector
from servo_controller import ServoController, DoorState
from waste_disposal_system import WasteDisposalSystem
import config
import utils

__all__ = [
    'PPEDetector',
    'ServoController', 
    'DoorState',
    'WasteDisposalSystem',
    'config',
    'utils'
]