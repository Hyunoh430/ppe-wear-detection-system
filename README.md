# 🥽 PPE Wear Detection System

## 📘 Project Overview

This project aims to enhance laboratory and industrial safety by detecting whether workers are properly wearing Personal Protective Equipment (PPE) such as goggles, masks, and gloves. It uses a Raspberry Pi with real-time camera input and integrates deep learning and pose estimation technologies to provide accurate and intelligent feedback.

---

## 🧠 Core Idea

Automatically determine whether PPE is being worn correctly using:

* **YOLOv8**: For object detection (goggles, masks, gloves)
* **MediaPipe**: For pose/keypoint estimation (eyes, nose, mouth, hands)
* **TTS / GUI / GPIO**: For real-time alerts and control actions

---

## 🧩 Technologies Used

* **YOLOv8 (Ultralytics)** for detecting PPE items in the frame
* **MediaPipe** for precise body keypoint tracking
* **OpenCV** for camera input and image processing
* **gTTS or pyttsx3** for audio alerts (TTS)
* **RPi.GPIO** for controlling hardware elements like a waste bin lock

---

## 🗂️ Folder Structure

```plaintext
ppe-wear-detection-system/
├── README.md
├── requirements.txt
├── models/
│   ├── yolov8n_ppe.pt
│   └── pose_landmark.tflite
├── src/
│   ├── main.py
│   ├── detector.py
│   ├── pose_estimator.py
│   ├── ppe_judger.py
│   ├── notifier.py
│   └── utils.py
├── data/
│   ├── sample_images/
│   └── inference_results/
├── scripts/
│   ├── convert_to_tflite.py
│   └── label_analysis.py
└── config/
    └── config.yaml
```

---

## ⚙️ How It Works

1. **Camera Input**: Captures real-time video stream from Pi camera or USB webcam
2. **YOLOv8 Detection**: Identifies bounding boxes for PPE and (optionally) human body
3. **MediaPipe Pose Estimation**: Extracts facial and hand keypoints
4. **Judgment Logic**: Checks whether PPE bounding boxes cover the appropriate body keypoints
5. **Smoothing & Decision**: Confirms consistent miswear before triggering warning
6. **Notifier**: Delivers voice, GUI, or GPIO control responses

---

## 📦 Setup & Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure the following packages are included:

* `ultralytics`
* `opencv-python`
* `mediapipe`
* `numpy`
* `gtts` or `pyttsx3`
* `RPi.GPIO` (for Raspberry Pi GPIO control)

---

## ▶️ Run the System

```bash
python src/main.py
```

---

## 📌 To-Do Checklist

* [v] Collect and annotate dataset for goggles, mask, gloves
* [ ] With Mask & goggles dataset
* [ ] Train YOLOv8 model with custom PPE dataset
* [ ] Optimize MediaPipe inference for Pi
* [ ] Implement wear judgment logic
* [ ] Integrate TTS and GPIO control
* [ ] Test full pipeline in real environment

---


## 📂 DATASETS



