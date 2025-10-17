# Gesture Control for PowerPoint
# Author: AJ Saludares
# Description: Performs real-time gesture detection and PowerPoint slide control using a trained LSTM model.
# Date: 15 May 2025

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from tensorflow.keras.models import load_model

# --- Configuration ---
MODEL_PATH = 'gesture_model_hand_swipe.h5' # Ensure this model exists or update path
actions = np.array(['swipe_left', 'swipe_right', 'neutral'])
threshold = 0.80
gesture_cooldown = 1.5 # seconds between gestures
last_gesture_time = 0
sequence_buffer = []

# --- PowerPoint Control Configuration ---
PPT_WINDOW_TITLE_SUBSTRING = "PowerPoint"  # Common substring in PowerPoint window titles
ppt_window = None

# --- Camera setup ---
cap = cv2.VideoCapture(0)
# Request higher FPS (camera must support it)
# cap.set(cv2.CAP_PROP_FPS, 60) # Uncomment if your camera supports it and you need higher FPS
# Optionally lower resolution for performance:
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# --- Load model ---
try:
    model = load_model(MODEL_PATH)
    seq_len, num_features = model.input_shape[1], model.input_shape[2]
    print(f"Loaded model: {MODEL_PATH}")
    print(f"Sequence length: {seq_len}; features: {num_features}")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please ensure '{MODEL_PATH}' is present and a valid Keras model.")
    exit()

# --- MediaPipe Hands setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Function to find and store the PowerPoint window ---
def find_and_set_ppt_window():
    global ppt_window
    try:
        windows = pyautogui.getWindowsWithTitle(PPT_WINDOW_TITLE_SUBSTRING)
        if windows:
            ppt_window = windows[0]  # Take the first match
            print(f"PowerPoint window found: '{ppt_window.title}'")
            # Bring it to front once, maybe not necessary every time
            # but good for initial setup
            try:
                if not ppt_window.isActive:
                     ppt_window.activate()
                # ppt_window.show() # Ensures it's not minimized
                # ppt_window.maximize() # Optional: if you want to maximize it
                print(f"Activated PowerPoint window: {ppt_window.title}")
            except Exception as e_activate:
                print(f"Could not activate window: {e_activate}. It might be minimized or require admin rights.")
                ppt_window = None # Reset if activation fails
        else:
            print(f"No window with title containing '{PPT_WINDOW_TITLE_SUBSTRING}' found.")
            ppt_window = None
    except Exception as e:
        print(f"Error finding PowerPoint window: {e}")
        print("PyAutoGUI might require accessibility permissions on some OS (e.g., macOS).")
        ppt_window = None

# Attempt to find the PowerPoint window at startup
find_and_set_ppt_window()
if not ppt_window:
    print(f"Warning: PowerPoint window not initially found. Gestures will not control PPT until a window with '{PPT_WINDOW_TITLE_SUBSTRING}' in its title is available and detected.")

print("\nStarting Gesture Control for PPT (press 'q' to quit)")
print("Ensure your PowerPoint presentation window is open.")
if ppt_window:
    print(f"Currently targeting: {ppt_window.title}")
else:
    print(f"Searching for a window with '{PPT_WINDOW_TITLE_SUBSTRING}' in its title.")


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame, skipping.")
        time.sleep(0.1) # Avoid busy-looping if camera fails
        continue

    frame = cv2.flip(frame, 1)
    display = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Extract keypoints
    keypoints = np.zeros(num_features) # Default to zeros if no hand
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(display, lm, mp_hands.HAND_CONNECTIONS)
        # Ensure we extract the correct number of features (x, y per landmark)
        # The model expects num_features = num_landmarks * 2
        extracted_kps = []
        for p in lm.landmark:
            extracted_kps.extend([p.x, p.y])
        
        if len(extracted_kps) == num_features:
            keypoints = np.array(extracted_kps)
        else:
            # This case should ideally not happen if num_features is set correctly
            # based on model.input_shape[2] which is num_landmarks * 2
            # For MediaPipe hands, there are 21 landmarks, so num_features should be 42.
            # If your model was trained with a different number, adjust accordingly.
            # print(f"Warning: Mismatch in keypoints. Expected {num_features}, got {len(extracted_kps)}. Padding/truncating.")
            keypoints = np.array(extracted_kps[:num_features]) # Truncate
            if len(keypoints) < num_features: # Pad with zeros if too short
                 keypoints = np.pad(keypoints, (0, num_features - len(keypoints)), 'constant')


    sequence_buffer.append(keypoints)

    # When buffer is full, attempt prediction
    if len(sequence_buffer) == seq_len:
        now = time.time()
        if now - last_gesture_time >= gesture_cooldown:
            data = np.expand_dims(np.array(sequence_buffer), axis=0)
            
            # Ensure data shape matches model input
            if data.shape[1] != seq_len or data.shape[2] != num_features:
                print(f"Data shape mismatch! Expected: (1, {seq_len}, {num_features}), Got: {data.shape}")
                sequence_buffer.pop(0) # Slide buffer window and try again
                continue

            probs = model.predict(data, verbose=0)[0]
            idx = np.argmax(probs)
            conf = probs[idx]

            if conf >= threshold:
                action = actions[idx]
                print(f"{time.strftime('%H:%M:%S')} - Gesture: {action}, Confidence={conf:.2f}")

                if action in ['swipe_right', 'swipe_left']:
                    if not ppt_window or not ppt_window.title: # Check if window is still valid
                        print("PowerPoint window not set or lost. Attempting to find it...")
                        find_and_set_ppt_window()

                    if ppt_window:
                        try:
                            if not ppt_window.isActive: # Activate only if not already active
                                print(f"Activating window: {ppt_window.title}")
                                ppt_window.activate()
                                time.sleep(0.1) # Give a moment for window to focus

                            if action == 'swipe_right':
                                pyautogui.press('right')
                                print("Next page")
                            elif action == 'swipe_left':
                                pyautogui.press('left')
                                print("Previous Page")
                            
                            last_gesture_time = now # Update time only on successful action
                            sequence_buffer.clear() # Clear buffer after successful action

                        except pyautogui.PyAutoGUIException as pae:
                            print(f"PyAutoGUI error during action: {pae}")
                            print("Window might have been closed. Trying to re-find...")
                            ppt_window = None # Reset ppt_window
                            find_and_set_ppt_window()
                            sequence_buffer.clear() # Clear buffer to avoid re-triggering immediately
                        except Exception as e_action:
                            print(f"An unexpected error occurred during action: {e_action}")
                            ppt_window = None # Reset
                            find_and_set_ppt_window()
                            sequence_buffer.clear()
                    else:
                        print("Action recognized, but no PowerPoint window is targeted. Please open/select PowerPoint.")
                        sequence_buffer.clear() # Clear buffer even if no action taken on PPT
                else: # Neutral action
                    print(f"{time.strftime('%H:%M:%S')} - Neutral gesture detected.")
                    # Do not clear sequence buffer for neutral, let it slide
                    # to allow continuous detection without missing gesture starts
                    # unless you specifically want a cooldown after neutral too.
                    # For this setup, neutral doesn't reset cooldown or buffer immediately.
                    # last_gesture_time = now # If you want neutral to also enforce cooldown
            else: # Low confidence
                # print(f"Low confidence prediction: {actions[idx]} ({conf:.2f}). Ignoring.")
                pass # Let the buffer slide without action

        # Slide buffer window if no high-confidence action was taken and buffer is full
        # If an action *was* taken and buffer cleared, this won't pop from an empty list.
        if len(sequence_buffer) == seq_len :
             sequence_buffer.pop(0)


    # Display buffering status
    buffer_status = f"BUFFER {len(sequence_buffer)}/{seq_len}"
    cv2.putText(display, buffer_status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow('Gesture Control PPT', display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Gesture control stopped.")
