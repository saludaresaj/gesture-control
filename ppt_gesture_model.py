# Gesture Control for PowerPoint
# Author: A. Saludares
# Description: Defines, trains, and saves an LSTM model for gesture classification.
# Date: 15 May 2025

import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# --- Configuration ---
DATA_PATH = "gesture_data_hand_swipe" # Should match the path from data collection
actions = np.array(['swipe_left', 'swipe_right', 'neutral'])
num_classes = len(actions)

# sequence_length should match data collection.
# Number of features: 21 landmarks * 2 coordinates (x, y) = 42
# If you collected x,y,z it would be 21*3 = 63
# This needs to be consistent with how data was saved in collect_gesture_data.py
# Assuming 21*2 = 42 features from the collection script.
# Let's try to determine sequence_length and num_features from the data itself
# to be more robust.
first_action_path = os.path.join(DATA_PATH, actions[0])
if os.listdir(first_action_path):
    sample_sequence_path = os.path.join(first_action_path, os.listdir(first_action_path)[0])
    sample_data = np.load(sample_sequence_path)
    sequence_length = sample_data.shape[0]
    num_features = sample_data.shape[1]
    print(f"Detected from data: sequence_length={sequence_length}, num_features={num_features}")
else:
    print(f"Error: No data found in {first_action_path} to determine sequence_length and num_features.")
    print("Please collect data first using collect_gesture_data.py")
    exit()


# --- Load Data ---
sequences, labels = [], []
label_map = {label: num for num, label in enumerate(actions)}

print("\nLoading data...")
for action_idx, action in enumerate(actions):
    action_path = os.path.join(DATA_PATH, action)
    for sequence_file in os.listdir(action_path):
        if sequence_file.endswith(".npy"):
            res = np.load(os.path.join(action_path, sequence_file))
            if res.shape == (sequence_length, num_features): # Basic check
                sequences.append(res)
                labels.append(label_map[action])
            else:
                print(f"Skipping file {sequence_file} due to unexpected shape: {res.shape}")

if not sequences:
    print("Error: No sequences loaded. Ensure data was collected correctly.")
    exit()

X = np.array(sequences)
y = to_categorical(labels, num_classes=num_classes).astype(int)

print(f"Data loaded: X shape: {X.shape}, y shape: {y.shape}") # (num_samples, sequence_length, num_features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# --- Define LSTM Model ---
print("\nBuilding LSTM model...")
model = Sequential()
model.add(InputLayer(input_shape=(sequence_length, num_features))) # (e.g., 20, 42)
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu')) # Last LSTM layer before Dense
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax')) # num_classes = number of actions

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# --- Callbacks (Optional but Recommended) ---
log_dir = os.path.join('logs') # For TensorBoard
tb_callback = TensorBoard(log_dir=log_dir)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)


# --- Train Model ---
print("\nTraining model...")
# Adjust epochs and batch_size as needed. Start with fewer epochs to test.
EPOCHS = 200 # Start with ~100-200, can increase if needed
BATCH_SIZE = 32

history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test),
                    callbacks=[tb_callback, early_stopping_callback])

# --- Evaluate Model ---
print("\nEvaluating model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# --- Save Model ---
MODEL_NAME = 'gesture_model_hand_swipe.h5' # Keras H5 format
# For TensorFlow 2.x, model.save() can save in SavedModel format by default if no .h5 extension
# model.save('gesture_model_hand_swipe_tf_format') # SavedModel format
model.save(MODEL_NAME)
print(f"\nModel saved as {MODEL_NAME}")

print("\nTraining complete!")
print(f"To visualize training, run: tensorboard --logdir={log_dir}")