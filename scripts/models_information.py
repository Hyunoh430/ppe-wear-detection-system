import tensorflow as tf
import numpy as np

def inspect_tflite_model(model_path):
    """TFLite 모델의 상세 정보를 출력"""
    print("="*50)
    print(f"TFLite 모델 분석: {model_path}")
    print("="*50)
    
    try:
        # TFLite 인터프리터 로드
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # 입력 텐서 정보
        input_details = interpreter.get_input_details()
        print("\n입력 텐서 정보:")
        print("-" * 30)
        for i, input_detail in enumerate(input_details):
            print(f"입력 {i}:")
            print(f"  - 이름: {input_detail['name']}")
            print(f"  - 형태: {input_detail['shape']}")
            print(f"  - 데이터 타입: {input_detail['dtype']}")
            print(f"  - 양자화 파라미터: {input_detail.get('quantization_parameters', 'None')}")
        
        # 출력 텐서 정보
        output_details = interpreter.get_output_details()
        print("\n출력 텐서 정보:")
        print("-" * 30)
        for i, output_detail in enumerate(output_details):
            print(f"출력 {i}:")
            print(f"  - 이름: {output_detail['name']}")
            print(f"  - 형태: {output_detail['shape']}")
            print(f"  - 데이터 타입: {output_detail['dtype']}")
            print(f"  - 양자화 파라미터: {output_detail.get('quantization_parameters', 'None')}")
        
        # 모델 메타데이터
        print(f"\n모델 메타데이터:")
        print("-" * 30)
        print(f"입력 텐서 개수: {len(input_details)}")
        print(f"출력 텐서 개수: {len(output_details)}")
        
        # 테스트 추론
        print(f"\n테스트 추론:")
        print("-" * 30)
        
        # 더미 입력 생성
        input_shape = input_details[0]['shape']
        dummy_input = np.random.rand(*input_shape).astype(input_details[0]['dtype'])
        print(f"더미 입력 형태: {dummy_input.shape}")
        
        # 추론 실행
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        
        # 출력 결과 확인
        for i, output_detail in enumerate(output_details):
            output_data = interpreter.get_tensor(output_detail['index'])
            print(f"출력 {i} 형태: {output_data.shape}")
            print(f"출력 {i} 값 범위: [{np.min(output_data):.4f}, {np.max(output_data):.4f}]")
            
            # YOLOv8 출력 분석
            if len(output_data.shape) == 3:
                batch_size, num_detections, features = output_data.shape
                print(f"  - 배치 크기: {batch_size}")
                print(f"  - 탐지 개수: {num_detections}")
                print(f"  - 특성 개수: {features}")
                
                if features == 6:
                    print("  - 예상 형식: [x_center, y_center, width, height, confidence, class_id]")
                elif features == 4 + len(['mask_weared_incorrect', 'with_mask', 'without_mask', 'with_gloves', 'without_gloves', 'goggles_on']):
                    print("  - 예상 형식: [x, y, w, h, class_0_prob, class_1_prob, ...]")
        
        print(f"\n클래스 정보:")
        print("-" * 30)
        class_names = ['mask_weared_incorrect', 'with_mask', 'without_mask', 
                      'with_gloves', 'without_gloves', 'goggles_on']
        for i, class_name in enumerate(class_names):
            print(f"클래스 {i}: {class_name}")
        
    except FileNotFoundError:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")

def inspect_pytorch_model(model_path):
    """PyTorch 모델 정보 확인 (선택사항)"""
    print("\n" + "="*50)
    print(f"PyTorch 모델 분석: {model_path}")
    print("="*50)
    
    try:
        import torch
        from ultralytics import YOLO
        
        # YOLO 모델 로드
        model = YOLO(model_path)
        
        print(f"모델 타입: {type(model.model)}")
        print(f"클래스 개수: {len(model.names)}")
        print(f"클래스 이름: {model.names}")
        
        # 모델 요약 (가능한 경우)
        if hasattr(model.model, 'info'):
            model.model.info()
            
    except ImportError:
        print("PyTorch 또는 Ultralytics가 설치되지 않았습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    # TFLite 모델 경로 (실제 경로로 변경하세요)
    tflite_model_path = "models/best3_float32_v3.tflite"
    pytorch_model_path = "models/best3.pt"
    
    # TFLite 모델 분석
    inspect_tflite_model(tflite_model_path)
    
    # PyTorch 모델 분석 (선택사항)
    # inspect_pytorch_model(pytorch_model_path)