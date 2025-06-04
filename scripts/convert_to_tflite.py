from ultralytics import YOLO

# Load a trained YOLOv8n model
model = YOLO("models/best3.pt")  

# Export the model to TFLite format
model.export(format="tflite")
