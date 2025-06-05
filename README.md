# PPE Waste Disposal System

A Raspberry Pi-based Personal Protective Equipment (PPE) detection and waste disposal system. This system uses a YOLO model to detect mask, gloves, and goggles wearing in real-time, and only opens the waste disposal entrance when all protective equipment is properly worn.

## 🚀 Key Features

- **Real-time PPE Detection**: High-performance object detection using YOLOv8 TFLite model
- **Automatic Door Control**: Automatic opening/closing of waste disposal entrance via servo motor
- **Safety Features**: 3-second continuous PPE verification before door opening
- **Auto Timer**: Automatic door closing after 5 seconds
- **Real-time Monitoring**: Real-time logging of FPS, detection status, and door status
- **Object-Oriented Design**: Modular code structure for improved maintainability

## 📋 System Requirements

### Hardware
- Raspberry Pi 4 (recommended) or compatible model
- Raspberry Pi Camera Module
- SG90 Servo Motor (or compatible model)
- GPIO connection jumper wires

### Software
- Python 3.7+
- TensorFlow Lite
- OpenCV
- Picamera2
- RPi.GPIO

## 🛠 Installation

1. **Clone Repository**
```bash
git clone <repository-url>
cd ppe-waste-disposal-system
```

2. **Install Required Packages**
```bash
pip install tensorflow-lite opencv-python numpy picamera2 RPi.GPIO
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
ppe-waste-disposal-system/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration management
│   ├── ppe_detector.py          # PPE detection class
│   ├── servo_controller.py      # Servo motor control class
│   ├── waste_disposal_system.py # Main system class
│   └── utils.py                 # Utility functions
├── models/
│   └── best3_float32_v3.tflite  # YOLO model file
├── logs/                        # Log file storage
├── main.py                      # Main execution file
└── README.md                    # Project documentation
```

## 🎯 Usage

### Basic Execution
```bash
python main.py
```

### Execution with Options
```bash
# Debug mode
python main.py --debug

# Use custom model
python main.py --model path/to/model.tflite

# Specify log file
python main.py --log-file system.log

# Run component tests only
python main.py --test-only

# Check system requirements
python main.py --check-requirements
```

### Execution Process

1. **System Start**: Initialize all components
2. **PPE Detection Wait**: Real-time monitoring with camera
3. **Condition Check**: 
   - ✅ Mask worn (`with_mask`)
   - ✅ Gloves worn (`with_gloves`) 
   - ✅ Goggles worn (`goggles_on`)
   - ❌ No improper wearing
4. **Timer Start**: 3-second continuous PPE verification
5. **Door Opening**: Automatic opening when conditions are met
6. **Auto Closing**: Automatic door closing after 5 seconds

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

## 🛡 Safety Features

- **Emergency Stop**: `Ctrl+C` or `emergency_stop()` method
- **Auto Recovery**: Return to safe state when errors occur
- **Permission Check**: Verify GPIO and camera access permissions
- **Resource Cleanup**: Automatic resource release on system shutdown

## 🐛 Troubleshooting

### Permission Errors
```bash
# Grant GPIO permissions
sudo usermod -a -G gpio $USER

# Or run with sudo
sudo python main.py
```

### Camera Errors
```bash
# Check camera activation
sudo raspi-config
# -> Interface Options -> Camera -> Enable
```

### Model File Errors
- Verify correct TFLite file exists in `models/` directory
- Check file path and permissions

## 📈 Performance Optimization

Recommendations for optimal performance on Raspberry Pi:

1. **Model Optimization**: Use quantized TFLite models
2. **Resolution Adjustment**: Adjust camera resolution as needed
3. **CPU Overclocking**: Improve CPU performance within stable limits
4. **Memory Split**: Adjust GPU memory allocation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is distributed under the MIT License. See the LICENSE file for details.

## 👥 Development Team

- **PPE Safety Team** - Initial development and maintenance

## 🔮 Future Plans

- [ ] Web-based monitoring dashboard
- [ ] Multi-camera support
- [ ] Cloud logging integration
- [ ] Mobile app integration
- [ ] AI model performance improvements