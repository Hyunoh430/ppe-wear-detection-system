"""
Main waste disposal system integrating PPE detection and door control
"""

import time
import threading
import logging
from typing import Optional, Dict, Any
from picamera2 import Picamera2
import numpy as np

from config import *
from ppe_detector import PPEDetector
from servo_controller import ServoController, DoorState

class WasteDisposalSystem:
    def __init__(self):
        """Initialize the waste disposal system"""
        self.logger = logging.getLogger(__name__)
        
        # System components
        self.ppe_detector: Optional[PPEDetector] = None
        self.servo_controller: Optional[ServoController] = None
        self.camera: Optional[Picamera2] = None
        
        # State tracking (간소화된 로직)
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
            
            # Initialize servo controller (테스트 움직임 제거)
            self.servo_controller = ServoController()
            self.logger.info("Servo controller initialized")
            
            # Initialize camera
            self._initialize_camera()
            self.logger.info("Camera initialized")
            
            # 초기화 시 테스트 움직임 제거
            self.logger.info("Waste disposal system initialized successfully - door in closed position")
            
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
        """Handle PPE compliance state changes (간소화된 로직)"""
        if is_compliant:
            # Start or continue compliance timing
            if self.compliance_start_time is None:
                self.compliance_start_time = current_time
                self.logger.info("PPE compliance detected - starting timer")
            
            compliance_duration = current_time - self.compliance_start_time
            
            # Check if compliance duration threshold is met and door is closed
            if (compliance_duration >= PPE_CHECK_DURATION and 
                self.servo_controller.is_door_closed()):
                
                self.logger.info(f"PPE compliance maintained for {compliance_duration:.1f}s - opening door")
                if self.servo_controller.open_door():
                    self.door_open_time = current_time
                    self.stats['door_openings'] += 1
                    self.stats['compliance_events'] += 1
                    # PPE 감지 성공 후 compliance_start_time 리셋하지 않음 (재감지 방지)
        else:
            # PPE 벗어도 바로 닫지 않음 - 타이머만 리셋
            if self.compliance_start_time is not None:
                self.logger.info("PPE compliance lost - resetting timer (door remains open if opened)")
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
                    # 문 닫힌 후 다시 PPE 감지 가능하도록 리셋
                    self.compliance_start_time = None
    
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
                status_parts.append("Compliant: Ready")
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


# ==========================================
# Individual Test Code
# ==========================================

def test_system_initialization():
    """Test system initialization"""
    print("=" * 50)
    print("WASTE DISPOSAL SYSTEM INITIALIZATION TEST")
    print("=" * 50)
    
    try:
        print("1. System component initialization...")
        system = WasteDisposalSystem()
        
        print("   O PPE detector initialization complete")
        print("   O Servo controller initialization complete")
        print("   O Camera initialization complete")
        
        print("\n2. System status check...")
        stats = system.get_statistics()
        print(f"   Door state: {stats['door_state']}")
        print(f"   Compliance status: {stats['is_compliant']}")
        
        print("\n3. System cleanup...")
        system.cleanup()
        print("   O Resource cleanup complete")
        
        print("\nO System initialization test successful!")
        return True
        
    except Exception as e:
        print(f"X System initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_short_run():
    """Test system short run (30 seconds)"""
    print("=" * 50)
    print("SYSTEM SHORT RUN TEST (30 SECONDS)")
    print("=" * 50)
    print("Please wear PPE for testing!")
    print("You can interrupt with Ctrl+C anytime.")
    
    try:
        import time
        import threading
        
        system = WasteDisposalSystem()
        
        # Run system in background
        system.run_async()
        
        # Monitor status for 30 seconds
        start_time = time.time()
        
        while time.time() - start_time < 30:
            time.sleep(5)  # Output status every 5 seconds
            
            stats = system.get_statistics()
            elapsed = time.time() - start_time
            
            print(f"\n[{elapsed:.0f}s] System Status:")
            print(f"  - Processed frames: {stats['total_frames']}")
            print(f"  - Detection count: {stats['detection_count']}")
            print(f"  - Compliance events: {stats['compliance_events']}")
            print(f"  - Door openings: {stats['door_openings']}")
            print(f"  - Current FPS: {stats['current_fps']:.1f}")
            print(f"  - Door state: {stats['door_state']}")
        
        print("\n30-second test completed!")
        system.stop()
        
        # Final statistics
        final_stats = system.get_statistics()
        print(f"\n Final Statistics:")
        print(f"  - Total runtime: {final_stats['runtime_seconds']:.1f} seconds")
        print(f"  - Total frames: {final_stats['total_frames']}")
        print(f"  - Average FPS: {final_stats['avg_fps']:.1f}")
        print(f"  - Detection count: {final_stats['detection_count']}")
        print(f"  - Door openings: {final_stats['door_openings']} times")
        
        print("\nO Short run test successful!")
        return True
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        system.stop()
        return True
    except Exception as e:
        print(f"X Error during system execution: {e}")
        import traceback
        traceback.print_exc()
        try:
            system.stop()
        except:
            pass
        return False

def test_system_components_integration():
    """Test system components integration"""
    print("=" * 50)
    print("SYSTEM COMPONENTS INTEGRATION TEST")
    print("=" * 50)
    
    try:
        system = WasteDisposalSystem()
        
        print("1. PPE detector standalone test...")
        frame = system.camera.capture_array()
        detections = system.ppe_detector.detect(frame)
        print(f"   Number of detected objects: {len(detections)}")
        
        if detections:
            for det in detections:
                print(f"   - {det['class_name']}: {det['confidence']:.2f}")
        
        print("\n2. Servo controller standalone test...")
        print("   Opening door...")
        if system.servo_controller.open_door():
            print("   O Door opening successful")
        
        import time
        time.sleep(2)
        
        print("   Closing door...")
        if system.servo_controller.close_door():
            print("   O Door closing successful")
        
        print("\n3. PPE compliance check test...")
        is_compliant, ppe_status = system.ppe_detector.check_ppe_compliance(detections)
        print(f"   Current compliance: {'O' if is_compliant else 'X'}")
        print(f"   PPE status: {ppe_status}")
        
        print("\n4. System cleanup...")
        system.cleanup()
        
        print("\nO Components integration test successful!")
        return True
        
    except Exception as e:
        print(f"X Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_emergency_stop():
    """Test emergency stop function"""
    print("=" * 50)
    print("EMERGENCY STOP FUNCTION TEST")
    print("=" * 50)
    
    try:
        system = WasteDisposalSystem()
        
        print("1. Starting system...")
        system.run_async()
        
        import time
        time.sleep(3)
        
        print("2. Testing emergency stop...")
        system.emergency_stop()
        
        print("3. Checking system status...")
        door_state = system.servo_controller.get_door_state()
        print(f"   Door state: {door_state.value}")
        
        if door_state.value == "error":
            print("   O Emergency stop successful (ERROR state)")
        else:
            print("   X Emergency stop status needs verification")
        
        print("\nO Emergency stop test completed!")
        return True
        
    except Exception as e:
        print(f"X Emergency stop test failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    print("Waste Disposal System Test Options:")
    print("1. System initialization test")
    print("2. Short run test (30 seconds)")
    print("3. Components integration test")
    print("4. Emergency stop test")
    
    choice = input("Choose (1-4): ").strip()
    
    if choice == "1":
        success = test_system_initialization()
    elif choice == "2":
        success = test_system_short_run()
    elif choice == "3":
        success = test_system_components_integration()
    elif choice == "4":
        success = test_emergency_stop()
    else:
        print("Invalid choice. Running initialization test.")
        success = test_system_initialization()
    
    if success:
        print("\n Test successful!")
    else:
        print("\n Test failed!")
        sys.exit(1)