import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split


GESTURE_CLASSES = ["Swipe Left", "Swipe Right", "Circle"]
DATA_PATH = "gesture_data"
SEQUENCE_LENGTH = 30
FRAME_SIZE = (640, 480)



def draw_control_panel(frame, recording, mode):

    status_color = (225, 255, 0) if recording else (0, 0, 255)
    cv2.putText(frame, f"{mode} MODE: {'ON' if recording else 'OFF'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)


    instructions = [
        "Controls:",
        "'S' - Start/Pause",
        "'X' - Save & Reset (Collection)",
        "'Q' - Quit",
        "'C' - Change Mode"
    ]

    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (10, 70 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)



def collect_gesture_data():

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)


    for gesture in GESTURE_CLASSES:
        os.makedirs(os.path.join(DATA_PATH, gesture), exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

    gesture_label = input("Enter gesture name to record: ")
    if gesture_label not in GESTURE_CLASSES:
        print("Invalid gesture! Exiting...")
        return

    recording = False
    landmark_sequence = deque(maxlen=SEQUENCE_LENGTH)
    sequence_count = len(os.listdir(os.path.join(DATA_PATH, gesture_label)))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            recording = not recording
            print(f"Recording {'started' if recording else 'paused'}")
        elif key == ord('x') and recording:
            if len(landmark_sequence) == SEQUENCE_LENGTH:
                save_path = os.path.join(DATA_PATH, gesture_label, f"{sequence_count}.npy")
                np.save(save_path, np.array(landmark_sequence))
                sequence_count += 1
                landmark_sequence.clear()
                print(f"Saved sequence {sequence_count} for {gesture_label}")
        elif key == ord('q'):
            break


        if recording and result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmark_sequence.append(landmarks)


                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display info
        draw_control_panel(frame, recording, "COLLECTION")
        cv2.putText(frame, f"Gesture: {gesture_label}", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Sequences saved: {sequence_count}", (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Gesture Data Collection", frame)

    cap.release()
    cv2.destroyAllWindows()


#----
def train_model():

    print("Loading data...")
    X, y = [], []
    label_map = {gesture: i for i, gesture in enumerate(GESTURE_CLASSES)}

    for gesture in GESTURE_CLASSES:
        gesture_dir = os.path.join(DATA_PATH, gesture)
        if not os.path.exists(gesture_dir):
            continue

        for seq_file in os.listdir(gesture_dir):
            sequence = np.load(os.path.join(gesture_dir, seq_file))
            X.append(sequence)
            y.append(label_map[gesture])

    if not X:
        print("No training data found! Collect data first.")
        return

    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Building model...")
    model = Sequential([
        LSTM(64, return_sequences=True, activation="relu", input_shape=(X.shape[1], X.shape[2])),
        LSTM(128, return_sequences=True, activation="relu"),
        LSTM(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(len(GESTURE_CLASSES), activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    print("Training...")
    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

    model.save("gesture_lstm_model.h5")
    print("Model saved!")


#----
def recognize_gestures():

    try:
        model = tf.keras.models.load_model("gesture_lstm_model.h5")
    except:
        print("No trained model found! Train a model first.")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    recognition_active = False
    current_gesture = "None"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            recognition_active = not recognition_active
            sequence.clear()
            print(f"Recognition {'active' if recognition_active else 'paused'}")
        elif key == ord('q'):
            break

        if recognition_active and result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                sequence.append(landmarks)


                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if len(sequence) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(np.array(sequence), axis=0)
                prediction = model.predict(input_data, verbose=0)
                current_gesture = GESTURE_CLASSES[np.argmax(prediction)]


        draw_control_panel(frame, recognition_active, "RECOGNITION")
        cv2.putText(frame, f"Current Gesture: {current_gesture}", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Gesture Recognition", frame)

    cap.release()
    cv2.destroyAllWindows()


#---
def main():
    while True:
        print("\nGesture Recognition System")
        print("1. Collect Gesture Data")
        print("2. Train Model")
        print("3. Recognize Gestures")
        print("4. Exit")

        choice = input("Select mode (1-4): ")

        if choice == '1':
            collect_gesture_data()
        elif choice == '2':
            train_model()
        elif choice == '3':
            recognize_gestures()
        elif choice == '4':
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()