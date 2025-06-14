# PPE Wear Detection System

A Raspberry Pi-based Personal Protective Equipment (PPE) detection and waste disposal system. This system uses a YOLO model to detect mask, gloves, and goggles wearing in real-time, and only opens the waste disposal entrance when all protective equipment is properly worn.

## 🚀 Key Features

- **Real-time PPE Detection**: High-performance object detection using YOLOv8 TFLite model
- **Automatic Door Control**: Servo motor-controlled waste disposal entrance
- **Video Streaming**: Real-time video transmission and web streaming capabilities
- **Safety Compliance**: 3-second continuous PPE verification before access
- **Auto Timer**: Automatic door closing after 5 seconds
- **System Monitoring**: Real-time FPS, detection status, and door monitoring
- **Comprehensive Testing**: Built-in test suite for all components
- **Remote Access**: Raspberry Pi communication and data transmission
- **Object-Oriented Design**: Modular architecture for maintainability

## 📋 System Requirements

### Hardware
- Raspberry Pi 4 (recommended) or compatible model
- Raspberry Pi Camera Module
- SG90 Servo Motor (or compatible model)
- GPIO connection jumper wires

### Software
- Python 3.8
- TensorFlow Lite
- OpenCV
- Picamera2
- RPi.GPIO
- Flask (for web streaming)
- NumPy

## 🛠 Installation

1. **Clone Repository**
```bash
git clone https://github.com/Hyunoh430/ppe-wear-detection-system.git
cd ppe-wear-detection-system
```

2. **Install Required Packages**
```bash
pip install tensorflow-lite opencv-python numpy picamera2 RPi.GPIO flask
```

3. **Prepare Model File**
```bash
mkdir models
# Place best3_float32_v3.tflite file in models/ directory
```

4. **Hardware Connection**
- Connect servo motor to GPIO pin 2
- Connect and activate camera module

## 📁 Project Structure

```
ppe-wear-detection-system/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration settings
│   ├── ppe_detector.py          # PPE detection with YOLO model
│   ├── raspberry_pi_sender.py   # Raspberry Pi data transmission
│   ├── servo_controller.py      # Servo motor control
│   ├── utils.py                 # Utility functions and system checks
│   ├── video_send.py           # Video streaming functionality
│   ├── waste_disposal_system.py # Main integrated system
│   └── README.md               # Comprehensive testing guide
├── models/
│   └── best3_float32_v3.tflite  # YOLO model file
├── logs/                        # System logs (auto-generated)
└── README.md                    # Project documentation
```

## 🎯 Usage

### Main System Execution
```bash
cd src
python waste_disposal_system.py
```

### Individual Component Testing
```bash
# PPE Detection Test
python ppe_detector.py

# Servo Motor Test  
python servo_controller.py

# System Utilities Test
python utils.py

# Video Streaming Test
python video_send.py

# Raspberry Pi Communication Test
python raspberry_pi_sender.py
```

### System Test Options
When running `waste_disposal_system.py`, you'll see:
```
1. 시스템 초기화 테스트
2. 단기 실행 테스트 (30초)
3. 컴포넌트 통합 테스트
4. 종료
```

### Execution Process

1. **System Start**: Initialize all components
2. **PPE Detection Wait**: Real-time monitoring with camera
3. **Condition Check**: 
   - ✅ Mask worn (`with_mask`) (Incorrect mask ❌, Without mask ❌)
   - ✅ Gloves worn (`with_gloves`) 
   - ✅ Goggles worn (`goggles_on`)
   - ❌ No improper wearing
4. **Timer Start**: 3-second continuous PPE verification
5. **Door Opening**: Automatic opening when conditions are met
6. **Auto Closing**: Automatic door closing after 5 seconds

## 🧪 Testing and Validation

### Comprehensive Test Suite
The system includes a detailed testing framework. For complete testing instructions, see [`src/README.md`](src/README.md).

### Quick Test Scenarios
```bash
# 1. Hardware Environment Check
cd src && python utils.py  # Select option 6

# 2. PPE Detection Validation  
python ppe_detector.py     # Test with various PPE combinations

# 3. Servo Motor Function Test
python servo_controller.py # Test door open/close operations

# 4. Video Streaming Test
python video_send.py      # Verify camera and streaming
```

## ⚙️ Configuration Customization

You can adjust various settings in the `src/config.py` file:

```python
# PPE check duration (seconds)
PPE_CHECK_DURATION = 3.0

# Door open duration (seconds)  
DOOR_OPEN_DURATION = 5.0

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.3

# Servo motor angle settings
SERVO_CLOSED_ANGLE = 20   # Closed state
SERVO_OPEN_ANGLE = 120    # Open state

# Video streaming settings
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
STREAM_PORT = 5000
```

## 🔧 Hardware Connection

### Servo Motor Connection
```
Servo Motor    ->  Raspberry Pi
VCC (Red)      ->  5V (Pin 2)
GND (Brown)    ->  GND (Pin 6)  
Signal(Orange) ->  GPIO 2 (Pin 3)
```

### Camera Connection
- Connect Raspberry Pi Camera Module to CSI port
- Enable camera in `sudo raspi-config`

## 📊 Monitoring and Logging

The system logs the following information in real-time:

- **FPS**: Frames processed per second
- **Detection Results**: List of detected PPE with confidence scores
- **Door Status**: Open/closed/moving status
- **Compliance**: PPE wearing compliance status
- **Statistics**: Total frames, detection count, door opening count
- **Video Stream**: Real-time video transmission status

## 🛡 Safety Features

- **Emergency Stop**: `Ctrl+C` or `emergency_stop()` method
- **Auto Recovery**: Return to safe state when errors occur
- **Permission Check**: Verify GPIO and camera access permissions
- **Resource Cleanup**: Automatic resource release on system shutdown
- **System Health Monitoring**: Continuous component status checking

## 🐛 Troubleshooting

### Permission Errors
```bash
# Grant GPIO permissions
sudo usermod -a -G gpio $USER

# Or run with sudo
sudo python waste_disposal_system.py
```

### Camera Errors
```bash
# Check camera activation
sudo raspi-config
# -> Interface Options -> Camera -> Enable

# Test camera manually
libcamera-hello --timeout 2000
```

### Model File Errors
- Verify correct TFLite file exists in `models/` directory
- Check file path and permissions
- Ensure model file is not corrupted

### Video Streaming Issues
```bash
# Check network connectivity
ping <target-ip>

# Verify Flask installation
python -c "import flask; print(flask.__version__)"
```

## 📈 Performance Optimization

Recommendations for optimal performance on Raspberry Pi:

1. **Model Optimization**: Use quantized TFLite models
2. **Resolution Adjustment**: Adjust camera resolution as needed (default: 640x480)
3. **CPU Overclocking**: Improve CPU performance within stable limits
4. **Memory Management**: Monitor and optimize memory usage
5. **Network Optimization**: Use appropriate video compression for streaming

## 🌐 Network Features

### Video Streaming
- Real-time video streaming via Flask server
- Configurable stream quality and resolution
- Multiple client support

### Data Transmission
- Raspberry Pi to external system communication
- JSON-based data protocol
- Real-time status updates

## 👥 Development Team

- **HyunOh Hong, Jiseok Park** - Hanyang University

## 🔮 Future Plans

- [ ] Web-based monitoring dashboard
- [ ] Cloud logging integration
- [ ] Mobile app integration
- [ ] AI model performance improvements
- [ ] Multi-camera support
- [ ] Advanced analytics and reporting
- [ ] IoT platform integration