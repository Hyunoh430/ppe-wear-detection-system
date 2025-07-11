# 🧪 PPE 폐기물 처리 시스템 테스트 가이드

이 문서는 시스템의 각 모듈을 개별적으로 테스트하는 방법을 상세히 설명합니다.

## 📋 목차

1. [테스트 개요](#-테스트-개요)
2. [테스트 환경 준비](#-테스트-환경-준비)
3. [개별 모듈 테스트](#-개별-모듈-테스트)
4. [테스트 시나리오](#-테스트-시나리오)
5. [문제 해결](#-문제-해결)
6. [테스트 결과 해석](#-테스트-결과-해석)

---

## 🎯 테스트 개요

### 테스트 가능한 모듈
- **PPE 감지기** (`ppe_detector.py`) - AI 모델 및 카메라 테스트
- **서보모터 제어** (`servo_controller.py`) - 하드웨어 제어 테스트
- **유틸리티** (`utils.py`) - 시스템 환경 및 설정 테스트
- **통합 시스템** (`waste_disposal_system.py`) - 전체 시스템 테스트

### 테스트 목적
- ✅ 개별 컴포넌트 동작 확인
- ✅ 하드웨어 연결 상태 검증
- ✅ 설정 파일 및 환경 검증
- ✅ 성능 및 안정성 확인

---

## 🛠 테스트 환경 준비

### 1. 기본 요구사항
```bash
# 필수 하드웨어
- Raspberry Pi (권장: Pi 4)
- Pi Camera Module
- SG90 서보모터
- 점퍼 케이블

# 소프트웨어 환경
- Python 3.7+
- 필수 패키지 설치됨
- GPIO 접근 권한
```

### 2. 파일 구조 확인
```
프로젝트/
├── src/
│   ├── ppe_detector.py      ← 개별 실행 가능
│   ├── servo_controller.py  ← 개별 실행 가능
│   ├── utils.py            ← 개별 실행 가능
│   └── waste_disposal_system.py ← 개별 실행 가능
├── models/
│   └── best3_float32_v3.tflite
└── logs/ (자동 생성)
```

### 3. 권한 설정
```bash
# GPIO 권한 (필수)
sudo usermod -a -G gpio $USER

# 카메라 활성화
sudo raspi-config
# → Interface Options → Camera → Enable

# 재부팅 권장
sudo reboot
```

---

## 🔍 개별 모듈 테스트

## 1️⃣ PPE 감지기 테스트

### 실행 방법
```bash
cd src
python ppe_detector.py
```

### 테스트 옵션

#### 옵션 1: 전체 테스트 (카메라 포함)
```
선택 (1 또는 2): 1
```

**테스트 내용:**
- ✅ TFLite 모델 로드
- ✅ 카메라 초기화
- ✅ 10초간 실시간 PPE 감지
- ✅ 감지 결과 분석

**예상 출력:**
```
=======================================
PPE DETECTOR 개별 테스트
=======================================
1. 모델 로드 테스트...
   ✓ 모델 로드 성공!
   - 입력 크기: 640x640
   - 클래스 목록: ['mask_weared_incorrect', 'with_mask', ...]

2. 카메라 초기화...
   ✓ 카메라 초기화 성공!

3. 실시간 PPE 감지 테스트 (10초간)...
   다양한 PPE를 착용하고 테스트해보세요!

   [2.1s] FPS: 15.2
   감지됨: with_mask(0.92), with_gloves(0.88)
   PPE 상태:
     - 마스크: ✓
     - 장갑: ✓
     - 고글: ✗
     - 위반사항: ✓
   전체 준수: ✗ PPE 미착용/부적절

4. 테스트 완료!
   - 총 프레임: 152
   - 평균 FPS: 15.2
   ✓ PPE 감지기 정상 작동!
```

#### 옵션 2: 모델만 테스트 (카메라 없이)
```
선택 (1 또는 2): 2
```

**사용 상황:** 카메라 문제 시 모델 파일만 검증

---

## 2️⃣ 서보모터 제어 테스트

### 실행 방법
```bash
cd src
python servo_controller.py
```

### 테스트 옵션

#### 옵션 1: 기본 테스트 (자동)
```
선택 (1-4): 1
```

**테스트 내용:**
- ✅ 서보 초기화 (20도)
- ✅ 도어 열기 (120도)
- ✅ 3초 대기
- ✅ 도어 닫기 (20도)

**예상 동작:**
```
서보모터가 부드럽게 20° → 120° → 20°로 이동
```

#### 옵션 2: 수동 제어 테스트
```
선택 (1-4): 2
```

**제어 명령어:**
```
명령어 입력: o    # 도어 열기
명령어 입력: c    # 도어 닫기
명령어 입력: s    # 상태 확인
명령어 입력: t    # 전체 테스트
명령어 입력: q    # 종료
```

**상태 확인 출력:**
```
현재 상태:
  - 도어 상태: open
  - 현재 각도: 120.0°
  - 도어 열림: True
  - 도어 닫힘: False
```

#### 옵션 3: 각도 테스트
```
선택 (1-4): 3
```

**테스트 내용:** 0°, 30°, 60°, 90°, 120°, 150°, 180° 순차 이동

#### 옵션 4: GPIO 설정 확인
```
선택 (1-4): 4
```

**확인 항목:**
- RPi.GPIO 라이브러리
- GPIO 권한
- PWM 초기화

---

## 3️⃣ 유틸리티 테스트

### 실행 방법
```bash
cd src
python utils.py
```

### 테스트 옵션

#### 옵션 1: 로깅 시스템 테스트
```
선택 (1-6): 1
```

**확인 항목:**
- 콘솔 로깅 동작
- 파일 로깅 생성
- 로그 레벨별 출력

#### 옵션 2: 시스템 체크 테스트
```
선택 (1-6): 2
```

**확인 항목:**
```
1. GPIO 권한 체크...
   GPIO 접근: ✓

2. 카메라 권한 체크...
   카메라 접근: ✓

3. 시스템 요구사항 체크...
   ✓ All system requirements met
```

#### 옵션 6: 모든 테스트 실행 (추천)
```
선택 (1-6): 6
```

**전체 테스트 결과:**
```
📊 테스트 결과 요약
=======================================
로깅 시스템    : ✓ 성공
시스템 체크    : ✓ 성공
모델 검증     : ✓ 성공
성능 모니터    : ✓ 성공
사용법 안내    : ✓ 성공

총 5개 테스트 중 5개 성공
🎉 모든 테스트 성공!
```

---

## 4️⃣ 통합 시스템 테스트

### 실행 방법
```bash
cd src
python waste_disposal_system.py
```

### 테스트 옵션

#### 옵션 1: 시스템 초기화 테스트
```
선택 (1-4): 1
```

**확인 항목:**
- PPE 감지기 초기화
- 서보 컨트롤러 초기화
- 카메라 초기화
- 리소스 정리

#### 옵션 2: 단기 실행 테스트 (30초) - 추천
```
선택 (1-4): 2
```

**테스트 과정:**
1. 시스템 백그라운드 실행
2. 5초마다 상태 출력
3. 30초 후 자동 종료
4. 최종 통계 출력

**실시간 출력 예시:**
```
[10s] 시스템 상태:
  - 처리 프레임: 152
  - 감지 횟수: 23
  - 컴플라이언스 이벤트: 1
  - 도어 열림 횟수: 1
  - 현재 FPS: 15.2
  - 도어 상태: closed
```

#### 옵션 3: 컴포넌트 통합 테스트
```
선택 (1-4): 3
```

**단계별 테스트:**
1. PPE 감지기 단독 동작
2. 서보 컨트롤러 단독 동작  
3. PPE 컴플라이언스 체크
4. 시스템 통합 동작

---

## 🎯 테스트 시나리오

### 시나리오 1: 초기 환경 검증 (5분)

```bash
# 1단계: 기본 환경 확인
cd src && python utils.py
# 선택: 6 (모든 테스트)

# 2단계: GPIO 및 서보 확인
python servo_controller.py
# 선택: 4 (GPIO 확인) → 1 (기본 테스트)

# 3단계: 모델 파일 확인
python ppe_detector.py  
# 선택: 2 (모델만 테스트)
```

**성공 기준:** 모든 테스트가 ✓ 상태

### 시나리오 2: PPE 감지 기능 검증 (10분)

```bash
# PPE 준비물: 마스크, 장갑, 고글

# 1단계: PPE 없이 테스트
python ppe_detector.py
# 선택: 1 (전체 테스트)
# 예상: 감지 없음 또는 without_mask 등

# 2단계: 일부 PPE 착용 테스트  
# 마스크만 착용하고 실행
# 예상: with_mask 감지, 전체 준수 ✗

# 3단계: 모든 PPE 착용 테스트
# 마스크 + 장갑 + 고글 착용하고 실행
# 예상: 모든 PPE 감지, 전체 준수 ✓
```

### 시나리오 3: 전체 시스템 동작 검증 (15분)

```bash
# 1단계: 시스템 통합 테스트
python waste_disposal_system.py
# 선택: 3 (컴포넌트 통합)

# 2단계: 실제 동작 테스트
python waste_disposal_system.py
# 선택: 2 (30초 실행)

# 3단계: PPE 착용 시나리오
# - 처음 10초: PPE 없이 대기
# - 다음 10초: 모든 PPE 착용 (3초 유지)
# - 마지막 10초: 도어 열림 확인
```

**예상 동작:**
1. PPE 미착용 → 도어 닫힌 상태 유지
2. PPE 완전 착용 3초 → 도어 자동 열림
3. 5초 후 → 도어 자동 닫힘

---

## 🔧 문제 해결

### 자주 발생하는 문제들

#### 1. 권한 오류
```bash
# 증상
PermissionError: [Errno 13] Permission denied: '/dev/gpiomem'

# 해결방법
sudo usermod -a -G gpio $USER
sudo reboot

# 또는 임시로
sudo python servo_controller.py
```

#### 2. 모듈 import 오류
```bash
# 증상  
ModuleNotFoundError: No module named 'src.config'

# 해결방법: src 디렉토리 내에서 실행
cd src
python ppe_detector.py

# 또는 PYTHONPATH 설정
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

#### 3. 카메라 접근 오류
```bash
# 증상
RuntimeError: Camera is not enabled

# 해결방법
sudo raspi-config
# Interface Options → Camera → Enable → 재부팅
```

#### 4. 모델 파일 없음
```bash
# 증상
FileNotFoundError: models/best3_float32_v3.tflite

# 해결방법
mkdir models
# 모델 파일을 models/ 디렉토리에 복사
```

#### 5. TensorFlow Lite 오류
```bash
# 증상
ImportError: No module named 'tensorflow'

# 해결방법
pip install tensorflow
# 또는 Lite 버전만
pip install tflite-runtime
```

---

## 📊 테스트 결과 해석

### 성공적인 테스트 결과

#### PPE 감지기
```
✓ 모델 로드 성공!
✓ 카메라 초기화 성공!
✓ PPE 감지기 정상 작동!

FPS: 10-20 (정상 범위)
감지 정확도: 0.3 이상 (신뢰할 수 있음)
```

#### 서보모터
```
✓ 초기화 성공!
✓ 도어 열기 성공!
✓ 도어 닫기 성공!
✓ 서보모터 정상 작동!

각도 변화: 20° ↔ 120° (정상 범위)
```

#### 통합 시스템
```
✓ 시스템 초기화 완료
✓ 컴포넌트 통합 테스트 성공!

FPS: 10+ (정상)
컴플라이언스 감지: 정상
도어 제어: 정상
```

### 비정상 결과 및 대응

#### 낮은 FPS (< 5)
```
원인: CPU 부족, 모델 크기 과대
대응: 해상도 낮추기, 모델 최적화
```

#### 감지 오류
```
원인: 조명 부족, 카메라 각도
대응: 조명 개선, 카메라 위치 조정
```

#### 서보 오동작
```
원인: 전력 부족, 배선 문제
대응: 외부 전원, 배선 재확인
```

---

## 🚀 고급 테스트 팁

### 1. 성능 최적화 테스트
```bash
# CPU 사용률 모니터링
htop

# 메모리 사용량 확인
free -h

# FPS 최적화 테스트
python ppe_detector.py
# 다양한 해상도로 테스트
```

### 2. 로그 분석
```bash
# 로그 파일 실시간 확인
tail -f logs/waste_disposal_*.log

# 에러 로그만 필터링
grep "ERROR" logs/*.log
```

### 3. 배치 테스트
```bash
# 모든 모듈 순차 테스트 스크립트
#!/bin/bash
cd src

echo "=== PPE 감지기 테스트 ==="
echo "2" | python ppe_detector.py

echo "=== 서보모터 테스트 ==="  
echo "1" | python servo_controller.py

echo "=== 유틸리티 테스트 ==="
echo "6" | python utils.py

echo "=== 시스템 초기화 테스트 ==="
echo "1" | python waste_disposal_system.py
```

---

## 📝 테스트 체크리스트

### 초기 설정 확인
- [ ] Python 3.7+ 설치
- [ ] 필수 패키지 설치
- [ ] GPIO 권한 설정
- [ ] 카메라 활성화
- [ ] 모델 파일 준비
- [ ] 서보모터 배선

### 개별 모듈 테스트
- [ ] utils.py 전체 테스트 성공
- [ ] servo_controller.py 기본 테스트 성공
- [ ] ppe_detector.py 모델 테스트 성공
- [ ] ppe_detector.py 전체 테스트 성공 (PPE 착용)

### 통합 테스트
- [ ] waste_disposal_system.py 초기화 성공
- [ ] waste_disposal_system.py 통합 테스트 성공
- [ ] 30초 실행 테스트에서 도어 동작 확인

### 실제 동작 확인
- [ ] PPE 미착용 시 도어 닫힌 상태 유지
- [ ] PPE 완전 착용 3초 후 도어 열림
- [ ] 5초 후 도어 자동 닫힘
- [ ] 비상 정지 기능 동작

---

이제 체계적으로 각 모듈을 테스트할 수 있습니다! 문제 발생 시 이 가이드를 참조해서 단계별로 해결해보세요. 🎯