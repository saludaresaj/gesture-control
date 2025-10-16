# gesture-control

Controlling PowerPoint slides using hand gestures tracked via MediaPipe and classified by an LSTM neural network trained on hand keypoints over time.

---

# Gesture-Based Control of PowerPoint Presentations

This system enables contactless control of PowerPoint presentations through simple hand gestures.  
By combining **MediaPipe** for hand tracking and an **LSTM neural network** for gesture classification,  
it translates your hand movements into slide navigation commands.

---

## 1. Introduction

Traditional presentation control often requires the use of clickers or keyboards, which can interrupt natural body movement and audience engagement. This project explores an alternative method using computer vision and machine learning to achieve intuitive, contactless presentation control through hand gestures.  

The system captures a sequence of hand landmarks from a webcam, encodes their motion over time, and uses a trained LSTM model to classify the gesture. Once recognized, the corresponding command is executed using automated keyboard inputs to navigate through PowerPoint slides.

---

## 2. How it Works

The gesture classes are defined as:
- **Neutral:** No change.  
- **Swipe Right:** Advance to next slide.  
- **Swipe Left:** Return to previous slide.  

---

## 3. Project Files

The project consists of three main components:
- **`ppt_gesture_training.py`:** Collects gesture data using MediaPipe Hands and saves them as NumPy arrays.
- **`ppt_gesture_model.py`:** Trains an LSTM model to classify temporal hand motion sequences into gesture categories.
- **`ppt_gesture_main.py`:** Performs real-time gesture detection and sends commands to PowerPoint through PyAutoGUI.  

---

## 4. Requirements

To run the system, install the following dependencies:
```bash
pip install opencv-python mediapipe tensorflow pyautogui scikit-learn numpy
