# PPE Waste Disposal System

라즈베리파이 기반의 개인보호장비(PPE) 착용 감지 및 폐기물 처리 시스템입니다. YOLO 모델을 사용하여 마스크, 장갑, 고글 착용 여부를 실시간으로 감지하고, 모든 보호장비가 올바르게 착용되었을 때만 폐기물 처리 입구를 개방합니다.

## 🚀 주요 기능

- **실시간 PPE 감지**: YOLOv8 TFLite 모델을 사용한 고성능 객체 감지
- **자동 도어 제어**: 서보모터를 통한 폐기물 처리 입구 자동 개폐
- **안전 기능**: 3초간 지속적인 PPE 착용 확인 후 도어 개방
- **자동 타이머**: 5초 후 자동 도어 닫힘
- **실시간 모니터링**: FPS, 감지 상태, 도어 상태 실시간 로깅
- **객체지향 설계**: 모듈화된 코드 구조로 유지보수성 향상

## 📋 시스템 요구사항

### 하드웨어
- Raspberry Pi 4 (권장) 또는 호환 모델
- Raspberry Pi Camera Module
- SG90 서보모터 (또는 호환 모델)
- GPIO 연결용 점퍼 와이어

### 소프트웨어
- Python 3.7+
- TensorFlow Lite
- OpenCV
- Picamera2
- RPi.GPIO

## 🛠 설치 방법

1. **저장소 클론**
```bash
git clone <repository-url>
cd ppe-waste-disposal-system
```

2. **필요 패키지 설치**
```bash
pip install tensorflow-lite opencv-python numpy picamera2 RPi.GPIO
```

3. **모델 파일 준비**
```bash
mkdir models
# best3_float32_v3.tflite 파일을 models/ 디렉토리에 배치
```

4. **하드웨어 연결**
- 서보모터를 GPIO 2번 핀에 연결
- 카메라 모듈 연결 및 활성화

## 📁 프로젝트 구조

```
ppe-waste-disposal-system/
├── src/
│   ├── __init__.py              # 패키지 초기화
│   ├── config.py                # 설정 관리
│   ├── ppe_detector.py          # PPE 감지 클래스
│   ├── servo_controller.py      # 서보모터 제어 클래스
│   ├── waste_disposal_system.py # 메인 시스템 클래스
│   └── utils.py                 # 유틸리티 함수
├── models/
│   └── best3_float32_v3.tflite  # YOLO 모델 파일
├── logs/                        # 로그 파일 저장소
├── main.py                      # 메인 실행 파일
└── README.md                    # 프로젝트 문서
```

## 🎯 사용 방법

### 기본 실행
```bash
python main.py
```

### 옵션과 함께 실행
```bash
# 디버그 모드
python main.py --debug

# 커스텀 모델 사용
python main.py --model path/to/model.tflite

# 로그 파일 지정
python main.py --log-file system.log

# 컴포넌트 테스트만 실행
python main.py --test-only

# 시스템 요구사항 확인
python main.py --check-requirements
```

### 실행 과정

1. **시스템 시작**: 모든 컴포넌트 초기화
2. **PPE 감지 대기**: 카메라로 실시간 모니터링
3. **조건 확인**: 
   - ✅ 마스크 착용 (`with_mask`)
   - ✅ 장갑 착용 (`with_gloves`) 
   - ✅ 고글 착용 (`goggles_on`)
   - ❌ 부적절한 착용 없음
4. **타이머 시작**: 3초간 지속적인 PPE 착용 확인
5. **도어 개방**: 조건 만족 시 자동 개방
6. **자동 닫힘**: 5초 후 자동으로 도어 닫힘

## ⚙️ 설정 커스터마이징

`src/config.py` 파일에서 다양한 설정을 조정할 수 있습니다:

```python
# PPE 체크 지속 시간 (초)
PPE_CHECK_DURATION = 3.0

# 도어 개방 지속 시간 (초)  
DOOR_OPEN_DURATION = 5.0

# 감지 신뢰도 임계값
CONFIDENCE_THRESHOLD = 0.3

# 서보모터 각도 설정
SERVO_CLOSED_ANGLE = 20   # 닫힌 상태
SERVO_OPEN_ANGLE = 120    # 열린 상태
```

## 🔧 하드웨어 연결

### 서보모터 연결
```
서보모터    ->  라즈베리파이
VCC (빨강)  ->  5V (핀 2)
GND (갈색)  ->  GND (핀 6)  
Signal(주황) ->  GPIO 2 (핀 3)
```

### 카메라 연결
- Raspberry Pi Camera Module을 CSI 포트에 연결
- `sudo raspi-config`에서 카메라 활성화

## 📊 모니터링 및 로깅

시스템은 다음 정보를 실시간으로 로깅합니다:

- **FPS**: 초당 프레임 처리 수
- **감지 결과**: 인식된 PPE 목록과 신뢰도
- **도어 상태**: 열림/닫힘/움직임 상태
- **컴플라이언스**: PPE 착용 준수 상태
- **통계**: 총 프레임, 감지 횟수, 도어 개방 횟수

## 🛡 안전 기능

- **비상 정지**: `Ctrl+C` 또는 `emergency_stop()` 메서드
- **자동 복구**: 오류 발생 시 안전 상태로 복귀
- **권한 검사**: GPIO 및 카메라 접근 권한 확인
- **리소스 정리**: 시스템 종료 시 자동 리소스 해제

## 🐛 문제 해결

### 권한 오류
```bash
# GPIO 권한 부여
sudo usermod -a -G gpio $USER

# 또는 sudo로 실행
sudo python main.py
```

### 카메라 오류
```bash
# 카메라 활성화 확인
sudo raspi-config
# -> Interface Options -> Camera -> Enable
```

### 모델 파일 오류
- `models/` 디렉토리에 올바른 TFLite 파일이 있는지 확인
- 파일 경로와 권한 확인

## 📈 성능 최적화

라즈베리파이에서의 최적 성능을 위한 권장사항:

1. **모델 최적화**: 양자화된 TFLite 모델 사용
2. **해상도 조정**: 필요에 따라 카메라 해상도 조정
3. **CPU 오버클럭**: 안정적인 범위에서 CPU 성능 향상
4. **메모리 분할**: GPU 메모리 분할 조정

## 🤝 기여 방법

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.

## 👥 개발팀

- **PPE Safety Team** - 초기 개발 및 유지보수

## 🔮 향후 계획

- [ ] 웹 기반 모니터링 대시보드
- [ ] 다중 카메라 지원
- [ ] 클라우드 로깅 연동
- [ ] 모바일 앱 연동
- [ ] AI 모델 성능 개선