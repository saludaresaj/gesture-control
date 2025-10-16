# Gesture Control for PowerPoint
# Author: A. Saludares
# Description: Collects gesture data using MediaPipe for LSTM model training.
# Date: 15 May 2025

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Configuration ---
# Path to store the collected data
DATA_PATH = "gesture_data_hand_swipe"
# Actions (gestures) to collect
actions = np.array(['swipe_left', 'swipe_right', 'neutral'])
# Number of sequences (videos/recordings) for each action
num_sequences = 45  # Collect 30 examples for each gesture. More is better!
# Number of frames per sequence
sequence_length = 15  # Each gesture recording will be 20 frames long.

# Create directories if they don't exist
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)

print("Data Collection Setup:")
print(f" - Data will be saved in: {DATA_PATH}")
print(f" - Actions to collect: {actions}")
print(f" - Sequences per action: {num_sequences}")
print(f" - Frames per sequence: {sequence_length}")
print("-" * 30)
print("Instructions:")
print(" - Press 's' to start collecting the CURRENT action displayed on screen.")
print(" - Perform the gesture naturally when 'RECORDING...' shows up.")
print(" - The script will cycle through actions and sequences.")
print(" - Press 'q' to quit early.")
print("-" * 30)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Loop through actions
action_idx = 0
while action_idx < len(actions):
    action = actions[action_idx]
    sequence_num = 0
    
    # Check if data for this action is already sufficient
    action_path = os.path.join(DATA_PATH, action)
    existing_sequences = len([name for name in os.listdir(action_path) if name.endswith(".npy")])
    sequence_num = existing_sequences

    while sequence_num < num_sequences:
        # Display current action and sequence number
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.flip(frame, 1) # Selfie view
        
        # Show instructions on the frame
        cv2.putText(frame, f"Press 's' to collect: {action} - Seq: {sequence_num + 1}/{num_sequences}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Ensure your hand is clearly visible.",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Data Collection', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
        elif key == ord('s'):
            print(f"\nStarting collection for {action}, sequence {sequence_num + 1}")
            # Countdown before recording
            for i in range(3, 0, -1):
                ret_wait, frame_wait = cap.read()
                if not ret_wait: break
                frame_wait = cv2.flip(frame_wait, 1)
                cv2.putText(frame_wait, f"Get Ready! Recording {action} in {i}...", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Data Collection', frame_wait)
                cv2.waitKey(1000) # 1 second delay

            sequence_data = []
            for frame_num in range(sequence_length):
                ret_rec, frame_rec = cap.read()
                if not ret_rec:
                    print("Error: Failed to grab frame during recording.")
                    break
                
                frame_rec_flipped = cv2.flip(frame_rec, 1)
                
                # Process hand
                rgb_image = cv2.cvtColor(frame_rec_flipped, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_image)

                # Draw landmarks for feedback
                display_frame = frame_rec_flipped.copy()
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        # Extract keypoints (x, y for all 21 landmarks)
                        keypoints = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()
                        sequence_data.append(keypoints)
                        break # Assuming one hand
                else:
                    # If no hand detected, append array of zeros (or handle differently, e.g., skip frame or sequence)
                    # For simplicity, we ensure 21 landmarks * 2 coordinates = 42 features
                    sequence_data.append(np.zeros(21 * 2)) 

                cv2.putText(display_frame, f"RECORDING... {action} - Frame: {frame_num + 1}/{sequence_length}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Data Collection', display_frame)
                cv2.waitKey(30) # Adjust for desired frame rate during recording

            if len(sequence_data) == sequence_length:
                npy_path = os.path.join(DATA_PATH, action, f"{action}_{sequence_num}.npy")
                np.save(npy_path, np.array(sequence_data))
                print(f"Saved: {npy_path}")
                sequence_num += 1
            else:
                print(f"Recording failed or interrupted for {action}, sequence {sequence_num + 1}. Retrying this sequence.")
                # No increment of sequence_num, so it will retry.
    
    action_idx +=1
    if action_idx >= len(actions):
        print("\nAll actions collected!")
        break


print("Data collection finished.")
cap.release()
cv2.destroyAllWindows()