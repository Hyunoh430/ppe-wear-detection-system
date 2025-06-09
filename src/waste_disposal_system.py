"""
Main waste disposal system with keyboard-triggered smart inference control
"""

import time
import threading
import logging
from typing import Optional, Dict, Any
from picamera2 import Picamera2
import numpy as np
import select
import sys
import tty
import termios

from config import *
from ppe_detector import PPEDetector
from servo_controller import ServoController, DoorState

class KeyboardListener:
    """Non-blocking keyboard input listener"""
    
    def __init__(self):
        self.old_settings = None
        self.setup_terminal()
    
    def setup_terminal(self):
        """Setup terminal for non-blocking input"""
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.cbreak(sys.stdin.fileno())
        except:
            pass  # Windows나 다른 환경에서는 스킵
    
    def restore_terminal(self):
        """Restore terminal settings"""
        try:
            if self.old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        except:
            pass
    
    def has_input(self):
        """Check if keyboard input is available"""
        try:
            return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
        except:
            return False
    
    def get_char(self):
        """Get single character if available"""
        try:
            if self.has_input():
                return sys.stdin.read(1)
        except:
            pass
        return None

class WasteDisposalSystem:
    def __init__(self):
        """Initialize the waste disposal system"""
        self.logger = logging.getLogger(__name__)
        
        # System components
        self.ppe_detector: Optional[PPEDetector] = None
        self.servo_controller: Optional[ServoController] = None
        self.camera: Optional[Picamera2] = None
        
        # Keyboard control
        self.keyboard_listener = KeyboardListener()
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
            'detection_sessions': 0,  # 검출 세션 수 추가
        }
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components"""
        try:
            self.logger.info("Initializing waste disposal system with keyboard control...")
            
            # Initialize PPE detector (미리 로드)
            self.logger.info("Loading PPE detection model...")
            self.ppe_detector = PPEDetector()
            self.logger.info("PPE detector loaded and ready!")
            
            # Initialize servo controller
            self.servo_controller = ServoController()
            self.logger.info("Servo controller initialized")
            
            # Initialize camera
            self._initialize_camera()
            self.logger.info("Camera initialized")
            
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
            if char.lower() == ' ':  # 스페이스바
                if not self.detection_in_progress:
                    self.detection_requested = True
                    self.stats['detection_sessions'] += 1
                    self.logger.info("🔍 PPE detection started!")
                else:
                    self.logger.info("Detection already in progress...")
            elif char.lower() == 'q':  # Q키로 종료
                self.logger.info("Quit requested by user")
                self.stop_event.set()
            elif char.lower() == 'r':  # R키로 리셋
                self._reset_detection()
                self.logger.info("Detection session reset")
            elif char.lower() == 'h':  # H키로 도움말
                self._print_help()
    
    def _reset_detection(self):
        """Reset detection session"""
        self.detection_requested = False
        self.detection_in_progress = False
        self.compliance_start_time = None
        if self.servo_controller.is_door_open():
            self.servo_controller.close_door()
            self.door_open_time = None
    
    def _print_help(self):
        """Print help message"""
        print("\n" + "="*50)
        print("KEYBOARD CONTROLS:")
        print("  SPACE  - Start PPE detection")
        print("  R      - Reset detection session")
        print("  Q      - Quit system")
        print("  H      - Show this help")
        print("="*50)
    
    def _should_run_inference(self) -> tuple[bool, str]:
        """Determine if inference should run"""
        
        # 1. 문이 열린 상태면 추론 중단 (서보 안정성 우선)
        if self.servo_controller.is_door_open():
            return False, "Door open - servo stability mode"
        
        # 2. 문이 움직이는 중이면 추론 중단
        if self.servo_controller.get_door_state() == DoorState.MOVING:
            return False, "Door moving - waiting for completion"
        
        # 3. 사용자가 요청했고, 아직 진행 중이지 않을 때만
        if self.detection_requested and not self.detection_in_progress:
            self.detection_in_progress = True
            self.detection_requested = False  # 한 번만 시작
            return True, "User requested detection"
        
        # 4. 이미 진행 중이면 계속
        if self.detection_in_progress:
            return True, "Detection session active"
        
        # 5. 기본적으로 대기 상태
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
        
        # 🚀 빠른 PPE 검출 (모델 이미 로드됨)
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
        # 추론이 활성화된 상태에서만 처리
        if not result.get('inference_active', False):
            return
        
        is_compliant = result['is_compliant']
        
        if is_compliant:
            if self.compliance_start_time is None:
                self.compliance_start_time = current_time
                self.logger.info("✅ PPE compliance detected - starting timer")
            
            compliance_duration = current_time - self.compliance_start_time
            
            # 컴플라이언스 달성 시 문 열기
            if (compliance_duration >= PPE_CHECK_DURATION and 
                self.servo_controller.is_door_closed()):
                
                self.logger.info(f"🎉 PPE compliance maintained for {compliance_duration:.1f}s - opening door")
                if self.servo_controller.open_door():
                    self.door_open_time = current_time
                    self.stats['door_openings'] += 1
                    self.stats['compliance_events'] += 1
                    self.detection_in_progress = False  # 검출 세션 종료
                    self.logger.info("🚪 Door opened! Detection session completed.")
        else:
            if self.compliance_start_time is not None:
                self.logger.info("❌ PPE compliance lost - resetting timer")
                self.compliance_start_time = None
    
    def _handle_door_timeout(self, current_time: float):
        """Handle automatic door closing after timeout"""
        if (self.door_open_time is not None and 
            self.servo_controller.is_door_open()):
            
            door_open_duration = current_time - self.door_open_time
            
            # 경고 메시지
            if door_open_duration >= DOOR_OPEN_DURATION - 3 and door_open_duration < DOOR_OPEN_DURATION:
                remaining = DOOR_OPEN_DURATION - door_open_duration
                if int(remaining * 2) % 2 == 0:  # 0.5초마다
                    self.logger.warning(f"⚠️  Door will close in {remaining:.1f} seconds")
            
            # 자동 닫기
            if door_open_duration >= DOOR_OPEN_DURATION:
                self.logger.info(f"🚪 Door timeout ({door_open_duration:.1f}s) - closing door")
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
        
        # 더 자주 로그 (활성 시에만)
        log_interval = 0.5 if result.get('inference_active', False) else 2.0
        if current_time - self.last_log_time < log_interval:
            return
        
        self.last_log_time = current_time
        
        status_parts = []
        
        # FPS info
        status_parts.append(f"FPS: {self.fps:.1f}")
        
        # Door state
        door_state = self.servo_controller.get_door_state().value
        door_symbol = "🚪[OPEN]" if door_state == "open" else "🔒[CLOSED]"
        status_parts.append(f"Door: {door_symbol}")
        
        # Detection status
        if result.get('inference_active', False):
            inference_time = result.get('inference_time_ms', 0)
            status_parts.append(f"🔍 Detection: ACTIVE ({inference_time:.1f}ms)")
        else:
            status_parts.append(f"⏸️  Detection: {result.get('status_reason', 'PAUSED')}")
        
        # Compliance status
        if result.get('inference_active', False):
            if result['is_compliant']:
                if self.compliance_start_time:
                    duration = current_time - self.compliance_start_time
                    progress = min(duration / PPE_CHECK_DURATION * 100, 100)
                    status_parts.append(f"✅ Compliant: {duration:.1f}s/{PPE_CHECK_DURATION}s ({progress:.0f}%)")
                else:
                    status_parts.append("✅ Compliant: Ready")
            else:
                status_parts.append("❌ Non-compliant")
        
        # Detection summary
        if result['detections'] and result.get('inference_active', False):
            status_parts.append(f"Detected: {result['detection_summary']}")
        elif result.get('inference_active', False):
            status_parts.append("No detections")
        
        # Door timer
        if self.door_open_time:
            door_duration = current_time - self.door_open_time
            remaining = DOOR_OPEN_DURATION - door_duration
            status_parts.append(f"⏰ Auto-close: {remaining:.1f}s")
        
        self.logger.info(" | ".join(status_parts))
    
    def run(self):
        """Main system loop with keyboard control"""
        try:
            self.is_running = True
            self.stats['start_time'] = time.time()
            self.last_detection_time = time.time()
            self.last_log_time = time.time()
            
            self.logger.info("🚀 Waste disposal system started with keyboard control")
            self.logger.info(f"⚙️  PPE check duration: {PPE_CHECK_DURATION}s")
            self.logger.info(f"⚙️  Door open duration: {DOOR_OPEN_DURATION}s")
            self.logger.info("📋 Controls: SPACE=Start detection, R=Reset, Q=Quit, H=Help")
            self.logger.info("🔄 System ready - press SPACE to start PPE detection")
            
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
                    time.sleep(0.03)  # 활성 시 빠르게 (33 FPS)
                else:
                    time.sleep(0.1)   # 대기 시 느리게 (10 FPS)
        
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
        
        # Restore terminal
        self.keyboard_listener.restore_terminal()
        
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
            
            # Restore terminal
            self.keyboard_listener.restore_terminal()
            
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
        self.logger.warning("🚨 EMERGENCY STOP ACTIVATED")
        
        if self.servo_controller:
            self.servo_controller.emergency_stop()
        
        self.stop()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process single frame for PPE detection"""
        self.frame_count += 1
        self.stats['total_frames'] += 1
        
        # Check if inference should run
        should_run, reason = self._should_run_inference()
        
        if not should_run:
            if self.inference_active:
                self.logger.info(f"Pausing inference: {reason}")
                self.inference_active = False
                self.inference_paused_reason = reason
            
            # Return empty result
            return {
                'detections': [],
                'is_compliant': False,
                'ppe_status': {},
                'detection_summary': "Inference paused",
                'inference_paused': True,
                'pause_reason': reason
            }
        
        # Resume inference if it was paused
        if not self.inference_active:
            self.logger.info(f"Resuming inference: {reason}")
            self.inference_active = True
            self.inference_paused_reason = ""
        
        # Perform PPE detection
        detections = self.ppe_detector.detect(frame, CONFIDENCE_THRESHOLD)
        
        if detections:
            self.stats['detection_count'] += 1
        
        # Check PPE compliance
        is_compliant, ppe_status = self.ppe_detector.check_ppe_compliance(detections)
        
        return {
            'detections': detections,
            'is_compliant': is_compliant,
            'ppe_status': ppe_status,
            'detection_summary': self.ppe_detector.get_detection_summary(detections),
            'inference_paused': False,
            'pause_reason': ""
        }
    
    def _handle_compliance_state(self, result: Dict[str, Any], current_time: float):
        """Handle PPE compliance state changes"""
        # 추론이 중단된 상태면 컴플라이언스 처리 안함
        if result.get('inference_paused', False):
            return
        
        is_compliant = result['is_compliant']
        
        if is_compliant:
            if self.compliance_start_time is None:
                self.compliance_start_time = current_time
                self.logger.info("PPE compliance detected - starting timer")
            
            compliance_duration = current_time - self.compliance_start_time
            
            if (compliance_duration >= PPE_CHECK_DURATION and 
                self.servo_controller.is_door_closed()):
                
                self.logger.info(f"PPE compliance maintained for {compliance_duration:.1f}s - opening door")
                if self.servo_controller.open_door():
                    self.door_open_time = current_time
                    self.stats['door_openings'] += 1
                    self.stats['compliance_events'] += 1
                    self.logger.info("Door opened successfully! Inference will pause for servo stability.")
        else:
            if self.compliance_start_time is not None:
                self.logger.info("PPE compliance lost - resetting timer")
                self.compliance_start_time = None
    
    def _handle_door_timeout(self, current_time: float):
        """Handle automatic door closing after timeout"""
        if (self.door_open_time is not None and 
            self.servo_controller.is_door_open()):
            
            door_open_duration = current_time - self.door_open_time
            
            if door_open_duration >= DOOR_OPEN_DURATION - 1 and door_open_duration < DOOR_OPEN_DURATION:
                remaining = DOOR_OPEN_DURATION - door_open_duration
                if int(remaining * 10) % 5 == 0:
                    self.logger.warning(f"WARNING: Door will close in {remaining:.1f} seconds")
            
            if door_open_duration >= DOOR_OPEN_DURATION:
                self.logger.info(f"Door open timeout ({door_open_duration:.1f}s) - closing door")
                if self.servo_controller.close_door():
                    self.door_open_time = None
                    self.compliance_start_time = None
                    self.logger.info("Door closed successfully - inference will resume")
    
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
        
        if current_time - self.last_log_time < 1.0:
            return
        
        self.last_log_time = current_time
        
        status_parts = []
        
        # FPS info
        status_parts.append(f"FPS: {self.fps:.1f}")
        
        # Door state
        door_state = self.servo_controller.get_door_state().value
        door_symbol = "[OPEN]" if door_state == "open" else "[CLOSED]"
        status_parts.append(f"Door: {door_symbol}")
        
        # Inference status
        if result.get('inference_paused', False):
            status_parts.append(f"Inference: PAUSED ({result['pause_reason']})")
        else:
            status_parts.append("Inference: ACTIVE")
        
        # Compliance status
        if not result.get('inference_paused', False) and result['is_compliant']:
            if self.compliance_start_time:
                duration = current_time - self.compliance_start_time
                progress = min(duration / PPE_CHECK_DURATION * 100, 100)
                status_parts.append(f"Compliant: {duration:.1f}s/{PPE_CHECK_DURATION}s ({progress:.0f}%)")
            else:
                status_parts.append("Compliant: Ready")
        elif not result.get('inference_paused', False):
            status_parts.append("Non-compliant")
        
        # Detection summary
        if result['detections'] and not result.get('inference_paused', False):
            status_parts.append(f"Detected: {result['detection_summary']}")
        elif not result.get('inference_paused', False):
            status_parts.append("No detections")
        
        # Door open timer
        if self.door_open_time:
            door_duration = current_time - self.door_open_time
            remaining = DOOR_OPEN_DURATION - door_duration
            status_parts.append(f"Door closes in: {remaining:.1f}s")
        
        self.logger.info(" | ".join(status_parts))
    
    def run(self):
        """Main system loop with smart inference control"""
        try:
            self.is_running = True
            self.stats['start_time'] = time.time()
            self.last_detection_time = time.time()
            self.last_log_time = time.time()
            
            self.logger.info("Starting waste disposal system with smart inference control")
            self.logger.info(f"PPE check duration: {PPE_CHECK_DURATION}s")
            self.logger.info(f"Door open duration: {DOOR_OPEN_DURATION}s")
            if hasattr(self, 'button_pin'):
                self.logger.info(f"Button control: GPIO {self.button_pin}")
            self.logger.info("Press Ctrl+C to stop")
            self.logger.info("System ready - waiting for PPE detection trigger...")
            
            while not self.stop_event.is_set():
                current_time = time.time()
                
                # Always capture frame (low cost)
                frame = self.camera.capture_array()
                
                # Smart processing (may skip inference)
                result = self._process_frame(frame)
                
                # Handle compliance state (only if inference active)
                self._handle_compliance_state(result, current_time)
                
                # Handle door timeout
                self._handle_door_timeout(current_time)
                
                # Update FPS
                self._update_fps(current_time)
                
                # Log status
                self._log_status(result, current_time)
                
                # Smart delay: longer when inference paused
                if result.get('inference_paused', False):
                    time.sleep(0.2)  # 추론 중단 시 더 느리게
                else:
                    time.sleep(0.05)  # 추론 활성 시 빠르게
        
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception as e:
            self.logger.error(f"System error: {e}")
        finally:
            self.stop()
    
    def request_manual_start(self):
        """Manually request PPE detection (for testing)"""
        self.manual_start_requested = True
        self.logger.info("Manual PPE detection requested via API")
    
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
            
            # Clean up button
            try:
                GPIO.remove_event_detect(self.button_pin)
            except:
                pass
            
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
            'inference_active': self.inference_active,
            'inference_paused_reason': self.inference_paused_reason
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