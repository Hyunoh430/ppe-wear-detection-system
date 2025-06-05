"""
Main waste disposal system integrating PPE detection and door control
"""

import time
import threading
import logging
from typing import Optional, Dict, Any
from picamera2 import Picamera2
import numpy as np

from .config import *
from .ppe_detector import PPEDetector
from .servo_controller import ServoController, DoorState

class WasteDisposalSystem:
    def __init__(self):
        """Initialize the waste disposal system"""
        self.logger = logging.getLogger(__name__)
        
        # System components
        self.ppe_detector: Optional[PPEDetector] = None
        self.servo_controller: Optional[ServoController] = None
        self.camera: Optional[Picamera2] = None
        
        # State tracking
        self.is_running = False
        self.compliance_start_time: Optional[float] = None
        self.door_open_time: Optional[float] = None
        self.last_detection_time = 0
        self.frame_count = 0
        self.fps = 0
        
        # Threading
        self.main_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'detection_count': 0,
            'compliance_events': 0,
            'door_openings': 0,
            'avg_fps': 0,
            'start_time': None
        }
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components"""
        try:
            self.logger.info("Initializing waste disposal system...")
            
            # Initialize PPE detector
            self.ppe_detector = PPEDetector()
            self.logger.info("PPE detector initialized")
            
            # Initialize servo controller
            self.servo_controller = ServoController()
            self.logger.info("Servo controller initialized")
            
            # Initialize camera
            self._initialize_camera()
            self.logger.info("Camera initialized")
            
            # Test servo movement
            if not self.servo_controller.test_movement():
                raise Exception("Servo test failed")
            
            self.logger.info("Waste disposal system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            self.cleanup()
            raise
    
    def _initialize_camera(self):
        """Initialize Picamera2"""
        try:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": CAMERA_FORMAT}
            )
            self.camera.configure(config)
            self.camera.start()
            
            # Wait for camera to stabilize
            time.sleep(2)
            
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            raise
    
    def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process single frame for PPE detection"""
        self.frame_count += 1
        self.stats['total_frames'] += 1
        
        # Perform PPE detection
        detections = self.ppe_detector.detect(frame, CONFIDENCE_THRESHOLD)
        
        if detections:
            self.stats['detection_count'] += 1
        
        # Check PPE compliance
        is_compliant, ppe_status = self.ppe_detector.check_ppe_compliance(detections)
        
        # Create frame result
        result = {
            'detections': detections,
            'is_compliant': is_compliant,
            'ppe_status': ppe_status,
            'detection_summary': self.ppe_detector.get_detection_summary(detections)
        }
        
        return result
    
    def _handle_compliance_state(self, is_compliant: bool, current_time: float):
        """Handle PPE compliance state changes"""
        if is_compliant:
            # Start or continue compliance timing
            if self.compliance_start_time is None:
                self.compliance_start_time = current_time
                self.logger.info("PPE compliance detected - starting timer")
            
            compliance_duration = current_time - self.compliance_start_time
            
            # Check if compliance duration threshold is met
            if (compliance_duration >= PPE_CHECK_DURATION and 
                self.servo_controller.is_door_closed()):
                
                self.logger.info(f"PPE compliance maintained for {compliance_duration:.1f}s - opening door")
                if self.servo_controller.open_door():
                    self.door_open_time = current_time
                    self.stats['door_openings'] += 1
                    self.stats['compliance_events'] += 1
        else:
            # Reset compliance timing
            if self.compliance_start_time is not None:
                self.logger.info("PPE compliance lost - resetting timer")
                self.compliance_start_time = None
    
    def _handle_door_timeout(self, current_time: float):
        """Handle automatic door closing after timeout"""
        if (self.door_open_time is not None and 
            self.servo_controller.is_door_open()):
            
            door_open_duration = current_time - self.door_open_time
            
            if door_open_duration >= DOOR_OPEN_DURATION:
                self.logger.info(f"Door open timeout ({door_open_duration:.1f}s) - closing door")
                if self.servo_controller.close_door():
                    self.door_open_time = None
    
    def _update_fps(self, current_time: float):
        """Update FPS calculation"""
        if self.frame_count % FPS_UPDATE_INTERVAL == 0:
            time_diff = current_time - self.last_detection_time
            if time_diff > 0:
                self.fps = FPS_UPDATE_INTERVAL / time_diff
                
                # Update average FPS
                if self.stats['start_time']:
                    total_time = current_time - self.stats['start_time']
                    self.stats['avg_fps'] = self.stats['total_frames'] / total_time
            
            self.last_detection_time = current_time
    
    def _log_status(self, result: Dict[str, Any], current_time: float):
        """Log current system status"""
        if not LOG_DETECTIONS and not DEBUG_MODE:
            return
        
        status_parts = []
        
        # FPS info
        status_parts.append(f"FPS: {self.fps:.1f}")
        
        # Door state
        door_state = self.servo_controller.get_door_state().value
        status_parts.append(f"Door: {door_state}")
        
        # Compliance status
        if result['is_compliant']:
            if self.compliance_start_time:
                duration = current_time - self.compliance_start_time
                status_parts.append(f"Compliant: {duration:.1f}s/{PPE_CHECK_DURATION}s")
            else:
                status_parts.append("Compliant: Starting")
        else:
            status_parts.append("Non-compliant")
        
        # Detection summary
        if result['detections']:
            status_parts.append(f"Detected: {result['detection_summary']}")
        else:
            status_parts.append("No detections")
        
        # Door open timer
        if self.door_open_time:
            door_duration = current_time - self.door_open_time
            status_parts.append(f"Door open: {door_duration:.1f}s/{DOOR_OPEN_DURATION}s")
        
        self.logger.info(" | ".join(status_parts))
    
    def run(self):
        """Main system loop"""
        try:
            self.is_running = True
            self.stats['start_time'] = time.time()
            self.last_detection_time = time.time()
            
            self.logger.info("Starting waste disposal system main loop")
            self.logger.info(f"PPE check duration: {PPE_CHECK_DURATION}s")
            self.logger.info(f"Door open duration: {DOOR_OPEN_DURATION}s")
            self.logger.info("Press Ctrl+C to stop")
            
            while not self.stop_event.is_set():
                current_time = time.time()
                
                # Capture frame
                frame = self.camera.capture_array()
                
                # Process frame
                result = self._process_frame(frame)
                
                # Handle compliance state
                self._handle_compliance_state(result['is_compliant'], current_time)
                
                # Handle door timeout
                self._handle_door_timeout(current_time)
                
                # Update FPS
                self._update_fps(current_time)
                
                # Log status
                if self.frame_count % FPS_UPDATE_INTERVAL == 0:
                    self._log_status(result, current_time)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception as e:
            self.logger.error(f"System error: {e}")
        finally:
            self.stop()
    
    def run_async(self):
        """Run system in separate thread"""
        if self.is_running:
            self.logger.warning("System already running")
            return
        
        self.main_thread = threading.Thread(target=self.run, daemon=True)
        self.main_thread.start()
        self.logger.info("System started in background thread")
    
    def stop(self):
        """Stop the system"""
        self.logger.info("Stopping waste disposal system...")
        self.stop_event.set()
        self.is_running = False
        
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=5)
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up system resources"""
        try:
            # Close door if open
            if self.servo_controller and self.servo_controller.is_door_open():
                self.servo_controller.close_door()
            
            # Clean up servo
            if self.servo_controller:
                self.servo_controller.cleanup()
            
            # Stop camera
            if self.camera:
                self.camera.stop()
            
            self.logger.info("System cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        current_time = time.time()
        runtime = current_time - self.stats['start_time'] if self.stats['start_time'] else 0
        
        return {
            **self.stats,
            'runtime_seconds': runtime,
            'current_fps': self.fps,
            'door_state': self.servo_controller.get_door_state().value if self.servo_controller else 'unknown',
            'is_compliant': self.compliance_start_time is not None,
            'compliance_duration': (current_time - self.compliance_start_time) if self.compliance_start_time else 0
        }
    
    def emergency_stop(self):
        """Emergency stop - immediately stop all operations"""
        self.logger.warning("EMERGENCY STOP ACTIVATED")
        
        if self.servo_controller:
            self.servo_controller.emergency_stop()
        
        self.stop()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()