import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import time


def text_detection_model_load():
    # Load the model and label map
    model = load_model("models/fine_tuned_alpha.h5")
    label_map = np.load("label_map/alphabet_label_map.npy", allow_pickle=True).item()

    # Reverse label map for prediction
    reverse_label_map = {v: k for k, v in label_map.items()}

    return model, reverse_label_map


def num_detection_model_load():
    # Load the model and label map
    model = load_model("models/asl_number_model_2.h5")
    label_map = np.load("label_map/num_label_map.npy", allow_pickle=True).item()

    # Reverse label map for prediction
    reverse_label_map = {v: k for k, v in label_map.items()}

    return model, reverse_label_map


def detection(model, reverse_label_map):
    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        try:
            data_aux = []
            x_ = []
            y_ = []
            z_ = []

            _, frame = cap.read()
            h, w, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        lm,
                        mp_hands.HAND_CONNECTIONS,
                        custom_landmark_style,
                        custom_connection_style,
                    )

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        z = hand_landmarks.landmark[i].z

                        x_.append(x)
                        y_.append(y)
                        z_.append(z)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        z = hand_landmarks.landmark[i].z

                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))
                        data_aux.append(z - min(z_))

                x1 = int(min(x_) * w) - 20
                y1 = int(min(y_) * h) - 20

                x2 = int(max(x_) * w) + 20
                y2 = int(max(y_) * h) + 20

                data_aux = np.array(data_aux).reshape(1, 21, 3, 1)
                prediction = model.predict(data_aux)
                predicted_class = reverse_label_map[np.argmax(prediction)]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
                cv2.putText(
                    frame,
                    predicted_class,
                    (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 140, 239),
                    4,
                    cv2.LINE_AA,
                )
                print(predicted_class)

            cv2.imshow("frame", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopping...")
                break
        except Exception as e:
            print(e)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Initialize Mediapipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    # Drawing utilities.
    mp_drawing = mp.solutions.drawing_utils

    # Custom styles for landmarks and connections.
    custom_landmark_style = mp_drawing.DrawingSpec(color=(244, 250, 197), thickness=4)
    custom_connection_style = mp_drawing.DrawingSpec(color=(250, 207, 97), thickness=2)

    time.sleep(2)
    choose_model = input(
        """
                         Choose one of the following model(1/2):
                            1. ASL Text Detection Model
                            2. ASL Number Detection
                         """
    )
    if choose_model == "1":
        model, reverse_label_map = text_detection_model_load()
        detection(model, reverse_label_map)

    elif choose_model == "2":
        model, reverse_label_map = num_detection_model_load()
        detection(model, reverse_label_map)
