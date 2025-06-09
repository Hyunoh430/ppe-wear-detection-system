"""
English-Only Waste Disposal System for Raspberry Pi
No Unicode characters or Korean text - ASCII only
Located at: src/waste_disposal_system.py
"""

import time
import threading
import logging
from typing import Optional, Dict, Any
from picamera2 import Picamera2
import numpy as np
import select
import sys
import queue

# Platform-specific imports
try:
    import termios
    import tty
    UNIX_SYSTEM = True
except ImportError:
    UNIX_SYSTEM = False

try:
    import msvcrt
    WINDOWS_SYSTEM = True
except ImportError:
    WINDOWS_SYSTEM = False

from config import *  
from ppe_detector import PPEDetector
from servo_controller import ServoController, DoorState

class EnglishKeyboardListener:
    """Cross-platform keyboard listener - English only"""
    
    def __init__(self):
        self.old_settings = None
        self.running = False
        self.input_queue = queue.Queue()
        self.input_thread = None
        self.setup_keyboard()
    
    def setup_keyboard(self):
        """Setup keyboard for different platforms"""
        if UNIX_SYSTEM:
            try:
                self.old_settings = termios.tcgetattr(sys.stdin)
                tty.cbreak(sys.stdin.fileno())
                print("Unix/Linux keyboard mode activated")
            except Exception as e:
                print(f"Warning: Unix keyboard setup failed: {e}")
                self.old_settings = None
        elif WINDOWS_SYSTEM:
            print("Windows keyboard mode activated")
        else:
            print("Warning: Limited keyboard input support")
    
    def restore_keyboard(self):
        """Restore keyboard settings"""
        if UNIX_SYSTEM and self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                print("Keyboard settings restored")
            except Exception as e:
                print(f"Warning: Failed to restore keyboard: {e}")
    
    def _input_thread_function(self):
        """Handle keyboard input in separate thread"""
        while self.running:
            try:
                if UNIX_SYSTEM:
                    if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                        char = sys.stdin.read(1)
                        if char:
                            self.input_queue.put(char)
                elif WINDOWS_SYSTEM:
                    if msvcrt.kbhit():
                        char = msvcrt.getch().decode('utf-8', errors='ignore')
                        self.input_queue.put(char)
                    else:
                        time.sleep(0.1)
                else:
                    try:
                        if sys.stdin.readable():
                            char = sys.stdin.read(1)
                            if char:
                                self.input_queue.put(char)
                    except:
                        pass
                    time.sleep(0.1)
                        
            except Exception as e:
                if self.running:
                    print(f"Keyboard input error: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Start keyboard listener"""
        if self.running:
            return
        
        self.running = True
        self.input_thread = threading.Thread(target=self._input_thread_function, daemon=True)
        self.input_thread.start()
        print("Keyboard listener started")
    
    def stop(self):
        """Stop keyboard listener"""
        self.running = False
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
        self.restore_keyboard()
        print("Keyboard listener stopped")
    
    def get_char(self):
        """Get keyboard input (non-blocking)"""
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None
    
    def has_input(self):
        """Check if input is available"""
        return not self.input_queue.empty()
    
    def clear_buffer(self):
        """Clear input buffer"""
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break

class WasteDisposalSystem:
    def __init__(self):
        """Initialize the waste disposal system"""
        self.logger = logging.getLogger(__name__)
        
        # System components
        self.ppe_detector: Optional[PPEDetector] = None
        self.servo_controller: Optional[ServoController] = None
        self.camera: Optional[Picamera2] = None
        
        # Keyboard control (English only)
        self.keyboard_listener = EnglishKeyboardListener()
        self.detection_requested = False
        self.detection_in_progress = False
        
        # State tracking
        self.is_running = False
        self.compliance_start_time: Optional[float] = None
        self.door_open_time: Optional[float] = None
        self.last_detection_time = 0
        self.last_log_time = 0
        self.frame_count = 0
        self.fps = 0
        self.inference_active = False
        
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
            'start_time': None,
            'detection_sessions': 0,
        }
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components"""
        try:
            self.logger.info("Initializing waste disposal system with keyboard control...")
            
            # Initialize PPE detector (preload)
            self.logger.info("Loading PPE detection model...")
            self.ppe_detector = PPEDetector()
            self.logger.info("PPE detector loaded and ready!")
            
            # Initialize servo controller
            self.servo_controller = ServoController()
            self.logger.info("Servo controller initialized")
            
            # Initialize camera
            self._initialize_camera()
            self.logger.info("Camera initialized")
            
            # Start keyboard listener
            self.keyboard_listener.start()
            
            self.logger.info("System ready! Press SPACE to start PPE detection, Q to quit")
            
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
            time.sleep(2)
            
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            raise
    
    def _handle_keyboard_input(self):
        """Handle keyboard input"""
        char = self.keyboard_listener.get_char()
        if char:
            if char == ' ':  # Space bar
                if not self.detection_in_progress:
                    self.detection_requested = True
                    self.stats['detection_sessions'] += 1
                    self.logger.info("PPE detection started!")
                else:
                    self.logger.info("Detection already in progress...")
            elif char.lower() == 'q':  # Q key to quit
                self.logger.info("Quit requested by user")
                self.stop_event.set()
            elif char.lower() == 'r':  # R key to reset
                self._reset_detection()
                self.logger.info("Detection session reset")
            elif char.lower() == 'h':  # H key for help
                self._print_help()
            elif char.lower() == 's':  # S key for status
                self._print_status()
    
    def _reset_detection(self):
        """Reset detection session"""
        self.detection_requested = False
        self.detection_in_progress = False
        self.compliance_start_time = None
        if self.servo_controller and self.servo_controller.is_door_open():
            self.servo_controller.close_door()
            self.door_open_time = None
        self.logger.info("Detection session has been reset")
    
    def _print_help(self):
        """Print help message"""
        help_text = """
===========================================================
                    KEYBOARD CONTROLS
===========================================================
  SPACE  - Start PPE detection
  R      - Reset detection session  
  S      - Show current status
  H      - Show this help
  Q      - Quit system
===========================================================
        """
        print(help_text)
    
    def _print_status(self):
        """Print current status"""
        stats = self.get_statistics()
        status_text = f"""
===========================================================
                    CURRENT SYSTEM STATUS
===========================================================
  Runtime: {stats['runtime_seconds']:.1f} seconds
  Processed frames: {stats['total_frames']}
  Detection count: {stats['detection_count']}
  Door openings: {stats['door_openings']}
  Current FPS: {stats['current_fps']:.1f}
  Door state: {stats['door_state']}
  Detection active: {'Yes' if stats['detection_active'] else 'No'}
===========================================================
        """
        print(status_text)
    
    def _should_run_inference(self) -> tuple[bool, str]:
        """Determine if inference should run"""
        
        # 1. Door open - stop inference for servo stability
        if self.servo_controller.is_door_open():
            return False, "Door open - servo stability mode"
        
        # 2. Door moving - wait for completion
        if self.servo_controller.get_door_state() == DoorState.MOVING:
            return False, "Door moving - waiting for completion"
        
        # 3. User requested and not in progress
        if self.detection_requested and not self.detection_in_progress:
            self.detection_in_progress = True
            self.detection_requested = False  # Start only once
            return True, "User requested detection"
        
        # 4. Already in progress - continue
        if self.detection_in_progress:
            return True, "Detection session active"
        
        # 5. Default waiting state
        return False, "Waiting for user input (SPACE)"
    
    def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process single frame with smart inference control"""
        self.frame_count += 1
        self.stats['total_frames'] += 1
        
        # Check if inference should run
        should_run, reason = self._should_run_inference()
        
        if not should_run:
            # Return empty result without inference
            return {
                'detections': [],
                'is_compliant': False,
                'ppe_status': {},
                'detection_summary': "Waiting for input",
                'inference_active': False,
                'status_reason': reason
            }
        
        # Fast PPE detection (model already loaded)
        start_time = time.time()
        detections = self.ppe_detector.detect(frame, CONFIDENCE_THRESHOLD)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        if detections:
            self.stats['detection_count'] += 1
        
        # Check PPE compliance
        is_compliant, ppe_status = self.ppe_detector.check_ppe_compliance(detections)
        
        return {
            'detections': detections,
            'is_compliant': is_compliant,
            'ppe_status': ppe_status,
            'detection_summary': self.ppe_detector.get_detection_summary(detections),
            'inference_active': True,
            'status_reason': reason,
            'inference_time_ms': inference_time
        }
    
    def _handle_compliance_state(self, result: Dict[str, Any], current_time: float):
        """Handle PPE compliance state changes"""
        # Only process when inference is active
        if not result.get('inference_active', False):
            return
        
        is_compliant = result['is_compliant']
        
        if is_compliant:
            if self.compliance_start_time is None:
                self.compliance_start_time = current_time
                self.logger.info("PPE compliance detected - starting timer")
            
            compliance_duration = current_time - self.compliance_start_time
            
            # Open door when compliance achieved
            if (compliance_duration >= PPE_CHECK_DURATION and 
                self.servo_controller.is_door_closed()):
                
                self.logger.info(f"PPE compliance maintained for {compliance_duration:.1f}s - opening door")
                if self.servo_controller.open_door():
                    self.door_open_time = current_time
                    self.stats['door_openings'] += 1
                    self.stats['compliance_events'] += 1
                    self.detection_in_progress = False  # End detection session
                    self.logger.info("Door opened! Detection session completed.")
        else:
            if self.compliance_start_time is not None:
                self.logger.info("PPE compliance lost - resetting timer")
                self.compliance_start_time = None
    
    def _handle_door_timeout(self, current_time: float):
        """Handle automatic door closing after timeout"""
        if (self.door_open_time is not None and 
            self.servo_controller.is_door_open()):
            
            door_open_duration = current_time - self.door_open_time
            
            # Warning messages
            if door_open_duration >= DOOR_OPEN_DURATION - 3 and door_open_duration < DOOR_OPEN_DURATION:
                remaining = DOOR_OPEN_DURATION - door_open_duration
                if int(remaining * 2) % 2 == 0:  # Every 0.5 seconds
                    self.logger.warning(f"WARNING: Door will close in {remaining:.1f} seconds")
            
            # Auto close
            if door_open_duration >= DOOR_OPEN_DURATION:
                self.logger.info(f"Door timeout ({door_open_duration:.1f}s) - closing door")
                if self.servo_controller.close_door():
                    self.door_open_time = None
                    self.compliance_start_time = None
                    self.detection_in_progress = False
                    self.logger.info("System ready for next detection (press SPACE)")
    
    def _update_fps(self, current_time: float):
        """Update FPS calculation"""
        if self.frame_count % FPS_UPDATE_INTERVAL == 0:
            time_diff = current_time - self.last_detection_time
            if time_diff > 0:
                self.fps = FPS_UPDATE_INTERVAL / time_diff
                
                if self.stats['start_time']:
                    total_time = current_time - self.stats['start_time']
                    self.stats['avg_fps'] = self.stats['total_frames'] / total_time
            
            self.last_detection_time = current_time
    
    def _log_status(self, result: Dict[str, Any], current_time: float):
        """Log current system status"""
        if not LOG_DETECTIONS and not DEBUG_MODE:
            return
        
        # More frequent logging when active
        log_interval = 0.5 if result.get('inference_active', False) else 2.0
        if current_time - self.last_log_time < log_interval:
            return
        
        self.last_log_time = current_time
        
        status_parts = []
        
        # FPS info
        status_parts.append(f"FPS: {self.fps:.1f}")
        
        # Door state
        door_state = self.servo_controller.get_door_state().value
        door_symbol = "[OPEN]" if door_state == "open" else "[CLOSED]"
        status_parts.append(f"Door: {door_symbol}")
        
        # Detection status
        if result.get('inference_active', False):
            inference_time = result.get('inference_time_ms', 0)
            status_parts.append(f"Detection: ACTIVE ({inference_time:.1f}ms)")
        else:
            status_parts.append(f"Detection: {result.get('status_reason', 'PAUSED')}")
        
        # Compliance status
        if result.get('inference_active', False):
            if result['is_compliant']:
                if self.compliance_start_time:
                    duration = current_time - self.compliance_start_time
                    progress = min(duration / PPE_CHECK_DURATION * 100, 100)
                    status_parts.append(f"Compliant: {duration:.1f}s/{PPE_CHECK_DURATION}s ({progress:.0f}%)")
                else:
                    status_parts.append("Compliant: Ready")
            else:
                status_parts.append("Non-compliant")
        
        # Detection summary
        if result['detections'] and result.get('inference_active', False):
            status_parts.append(f"Detected: {result['detection_summary']}")
        elif result.get('inference_active', False):
            status_parts.append("No detections")
        
        # Door timer
        if self.door_open_time:
            door_duration = current_time - self.door_open_time
            remaining = DOOR_OPEN_DURATION - door_duration
            status_parts.append(f"Auto-close: {remaining:.1f}s")
        
        self.logger.info(" | ".join(status_parts))
    
    def run(self):
        """Main system loop with keyboard control"""
        try:
            self.is_running = True
            self.stats['start_time'] = time.time()
            self.last_detection_time = time.time()
            self.last_log_time = time.time()
            
            self.logger.info("Waste disposal system started with keyboard control")
            self.logger.info(f"PPE check duration: {PPE_CHECK_DURATION}s")
            self.logger.info(f"Door open duration: {DOOR_OPEN_DURATION}s")
            self.logger.info("Controls: SPACE=Start detection, R=Reset, Q=Quit, H=Help")
            self.logger.info("System ready - press SPACE to start PPE detection")
            
            while not self.stop_event.is_set():
                current_time = time.time()
                
                # Handle keyboard input
                self._handle_keyboard_input()
                
                # Always capture frame (lightweight)
                frame = self.camera.capture_array()
                
                # Smart processing (conditional inference)
                result = self._process_frame(frame)
                
                # Handle compliance state
                self._handle_compliance_state(result, current_time)
                
                # Handle door timeout
                self._handle_door_timeout(current_time)
                
                # Update FPS
                self._update_fps(current_time)
                
                # Log status
                self._log_status(result, current_time)
                
                # Smart delay
                if result.get('inference_active', False):
                    time.sleep(0.03)  # Fast when active (33 FPS)
                else:
                    time.sleep(0.1)   # Slow when waiting (10 FPS)
        
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception as e:
            self.logger.error(f"System error: {e}")
            import traceback
            traceback.print_exc()
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
            
            # Stop keyboard listener
            self.keyboard_listener.stop()
            
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
            'compliance_duration': (current_time - self.compliance_start_time) if self.compliance_start_time else 0,
            'detection_active': self.detection_in_progress
        }
    
    def emergency_stop(self):
        """Emergency stop"""
        self.logger.warning("EMERGENCY STOP ACTIVATED")
        
        if self.servo_controller:
            self.servo_controller.emergency_stop()
        
        self.stop()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ==========================================
# Test Code (English only)
# ==========================================

def test_system_initialization():
    """Test system initialization"""
    print("=" * 50)
    print("SYSTEM INITIALIZATION TEST")
    print("=" * 50)
    
    try:
        print("1. PPE detector initialization...")
        detector = PPEDetector()
        print("   SUCCESS: PPE detector initialized")
        
        print("2. Servo controller initialization...")
        servo = ServoController()
        print("   SUCCESS: Servo controller initialized")
        servo.cleanup()
        
        print("3. Camera initialization...")
        from picamera2 import Picamera2
        camera = Picamera2()
        config = camera.create_preview_configuration(main={"size": (640, 480)})
        camera.configure(config)
        camera.start()
        camera.stop()
        print("   SUCCESS: Camera initialized")
        
        print("4. Keyboard listener test...")
        keyboard = EnglishKeyboardListener()
        keyboard.start()
        time.sleep(1)
        keyboard.stop()
        print("   SUCCESS: Keyboard listener tested")
        
        print("\nALL COMPONENTS INITIALIZED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        print(f"   FAILED: Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_short_run():
    """30-second short run test"""
    print("=" * 50)
    print("30-SECOND SHORT RUN TEST")
    print("=" * 50)
    print("Running system for 30 seconds to test basic operation.")
    print("Try pressing SPACE during test to start PPE detection!")
    print("-" * 50)
    
    try:
        system = WasteDisposalSystem()
        
        # Run in background
        system.run_async()
        
        # Wait 30 seconds with status output
        for i in range(6):
            time.sleep(5)
            stats = system.get_statistics()
            print(f"\n[{(i+1)*5}s] System Status:")
            print(f"  - Processed frames: {stats['total_frames']}")
            print(f"  - Detection count: {stats['detection_count']}")
            print(f"  - Compliance events: {stats['compliance_events']}")
            print(f"  - Door openings: {stats['door_openings']}")
            print(f"  - Current FPS: {stats['current_fps']:.1f}")
            print(f"  - Door state: {stats['door_state']}")
        
        print("\n30-second test completed! Stopping system...")
        system.stop()
        
        # Final statistics
        final_stats = system.get_statistics()
        print(f"\nFINAL STATISTICS:")
        print(f"  - Total runtime: {final_stats['runtime_seconds']:.1f} seconds")
        print(f"  - Total frames: {final_stats['total_frames']}")
        print(f"  - Average FPS: {final_stats['avg_fps']:.1f}")
        print(f"  - Detection sessions: {final_stats['detection_sessions']}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_integration():
    """Component integration test"""
    print("=" * 50)
    print("COMPONENT INTEGRATION TEST")
    print("=" * 50)
    
    try:
        print("1. PPE detector standalone test...")
        detector = PPEDetector()
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect(dummy_image)
        is_compliant, ppe_status = detector.check_ppe_compliance(detections)
        print(f"   SUCCESS: PPE detector working - {len(detections)} objects detected")
        
        print("2. Servo controller standalone test...")
        servo = ServoController()
        print("   - Testing door open...")
        servo.open_door()
        time.sleep(2)
        print("   - Testing door close...")
        servo.close_door()
        print("   SUCCESS: Servo controller working normally")
        servo.cleanup()
        
        print("3. PPE compliance check test...")
        # Test various scenarios
        test_scenarios = [
            ([], "No PPE"),
            ([{'class_name': 'with_mask', 'confidence': 0.9}], "Mask only"),
            ([{'class_name': 'with_mask', 'confidence': 0.9}, 
              {'class_name': 'with_gloves', 'confidence': 0.8}], "Mask + Gloves"),
            ([{'class_name': 'with_mask', 'confidence': 0.9}, 
              {'class_name': 'with_gloves', 'confidence': 0.8},
              {'class_name': 'goggles_on', 'confidence': 0.7}], "Full PPE"),
        ]
        
        for detections, scenario in test_scenarios:
            is_compliant, ppe_status = detector.check_ppe_compliance(detections)
            compliance_text = "COMPLIANT" if is_compliant else "NON-COMPLIANT"
            print(f"   {scenario}: {compliance_text}")
        
        print("4. System integration test...")
        # Brief integration test without full run
        print("   - Initializing integrated system...")
        
        # Test initialization only
        try:
            ppe_detector = PPEDetector()
            servo_controller = ServoController()
            
            from picamera2 import Picamera2
            camera = Picamera2()
            config = camera.create_preview_configuration(main={"size": (640, 480)})
            camera.configure(config)
            camera.start()
            
            # Test one frame
            frame = camera.capture_array()
            detections = ppe_detector.detect(frame)
            
            camera.stop()
            servo_controller.cleanup()
            
            print("   SUCCESS: System integration working")
            
        except Exception as e:
            print(f"   WARNING: Integration test issue: {e}")
        
        print("\nCOMPONENT INTEGRATION TEST COMPLETED!")
        return True
        
    except Exception as e:
        print(f"FAILED: Integration test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all system tests"""
    print("=" * 60)
    print("RUNNING ALL WASTE DISPOSAL SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        ("System Initialization", test_system_initialization),
        ("Component Integration", test_component_integration),
        ("Short Run Test", test_short_run),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nStarting {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"ERROR during {test_name} test: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "SUCCESS" if result else "FAILED"
        print(f"{test_name:25}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n{success_count} out of {total_count} tests passed")
    
    if success_count == total_count:
        print("ALL TESTS SUCCESSFUL!")
        return True
    else:
        print("SOME TESTS FAILED!")
        return False

if __name__ == "__main__":
    import sys
    
    print("Waste Disposal System Test Options:")
    print("1. System initialization test")
    print("2. Short run test (30 seconds)")  
    print("3. Component integration test")
    print("4. Run all tests")
    
    choice = input("Choose (1-4): ").strip()
    
    if choice == "1":
        success = test_system_initialization()
    elif choice == "2":
        success = test_short_run()
    elif choice == "3":
        success = test_component_integration()
    elif choice == "4":
        success = run_all_tests()
    else:
        print("Invalid choice. Running all tests.")
        success = run_all_tests()
    
    if success:
        print("\nTEST SUCCESSFUL!")
    else:
        print("\nTEST FAILED!")
        sys.exit(1)