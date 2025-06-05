#!/usr/bin/env python3
"""
Main entry point for the PPE Waste Disposal System
"""

import argparse
import signal
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.waste_disposal_system import WasteDisposalSystem
from src.utils import (
    setup_logging, 
    create_log_filename, 
    print_usage_instructions,
    signal_handler,
    validate_model_file,
    create_model_directory
)
from src.config import MODEL_PATH, DEBUG_MODE

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="PPE Waste Disposal System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run with default settings
  python main.py --debug                  # Run in debug mode
  python main.py --model custom.tflite    # Use custom model
  python main.py --log-file system.log    # Log to specific file
  python main.py --test-only               # Test components only
        """
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default=MODEL_PATH,
        help=f'Path to TFLite model file (default: {MODEL_PATH})'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    
    parser.add_argument(
        '--log-file', 
        type=str,
        help='Log file path (default: auto-generated timestamp)'
    )
    
    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--test-only', 
        action='store_true',
        help='Run component tests only, do not start main system'
    )
    
    parser.add_argument(
        '--no-log-file', 
        action='store_true',
        help='Disable file logging, only log to console'
    )
    
    return parser.parse_args()

def test_components(model_path: str) -> bool:
    """Test system components (간소화됨)"""
    print("\n" + "="*50)
    print("COMPONENT TESTING")
    print("="*50)
    
    success = True
    
    # Test 1: Model file validation
    print("1. Testing model file...")
    if not validate_model_file(model_path):
        print("   X Model file validation failed")
        success = False
    else:
        print("   O Model file validation passed")
    
    # Test 2: PPE Detector
    print("2. Testing PPE detector...")
    try:
        from src.ppe_detector import PPEDetector
        detector = PPEDetector(model_path)
        print("   O PPE detector initialization successful")
    except Exception as e:
        print(f"   X PPE detector failed: {e}")
        success = False
    
    # Test 3: Servo Controller (테스트 움직임 제거)
    print("3. Testing servo controller...")
    try:
        from src.servo_controller import ServoController
        servo = ServoController()
        print("   O Servo controller initialization successful")
        servo.cleanup()
    except Exception as e:
        print(f"   X Servo controller failed: {e}")
        success = False
    
    # Test 4: Camera
    print("4. Testing camera...")
    try:
        from picamera2 import Picamera2
        camera = Picamera2()
        config = camera.create_preview_configuration(main={"size": (640, 480)})
        camera.configure(config)
        camera.start()
        
        # Capture test frame
        frame = camera.capture_array()
        camera.stop()
        
        if frame is not None and frame.size > 0:
            print("   O Camera test successful")
        else:
            print("   X Camera test failed - invalid frame")
            success = False
            
    except Exception as e:
        print(f"   X Camera test failed: {e}")
        success = False
    
    print("="*50)
    if success:
        print("O ALL COMPONENT TESTS PASSED")
    else:
        print("X SOME COMPONENT TESTS FAILED")
    print("="*50)
    
    return success

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create necessary directories
    create_model_directory()
    
    # Setup logging
    log_file = None
    if not args.no_log_file:
        log_file = args.log_file or create_log_filename()
    
    log_level = 'DEBUG' if args.debug else args.log_level
    logger = setup_logging(log_level, log_file)
    
    # Validate model file
    if not validate_model_file(args.model):
        logger.error("Model validation failed")
        return 1
    
    # Run component tests if requested
    if args.test_only:
        success = test_components(args.model)
        return 0 if success else 1
    
    # Initialize and run system
    system = None
    try:
        logger.info("Starting PPE Waste Disposal System")
        
        # Print usage instructions
        print_usage_instructions()
        
        # Initialize system (더 이상 requirements 체크 안함)
        logger.info("Initializing system components...")
        system = WasteDisposalSystem()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler(system))
        signal.signal(signal.SIGTERM, signal_handler(system))
        
        # Run system
        logger.info("System initialized successfully - starting main loop")
        logger.info("Door is in closed position, ready for PPE detection")
        system.run()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        if system:
            # Print final statistics
            stats = system.get_statistics()
            logger.info("Final Statistics:")
            logger.info(f"  Runtime: {stats['runtime_seconds']:.1f} seconds")
            logger.info(f"  Total frames: {stats['total_frames']}")
            logger.info(f"  Detection events: {stats['detection_count']}")
            logger.info(f"  Compliance events: {stats['compliance_events']}")
            logger.info(f"  Door openings: {stats['door_openings']}")
            logger.info(f"  Average FPS: {stats['avg_fps']:.1f}")
            
            system.stop()
        
        logger.info("System shutdown complete")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())