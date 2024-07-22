import os
import numpy as np
import mediapipe as mp
import cv2


def dataProcessor(source, target, lmap, skip=[]):
    # Initialize Mediapipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

    # Define the data directory
    data_dir = f"Data/{source}"

    data = []
    labels = []
    label_map = {}

    # Process each image in the directory
    for dir_ in os.listdir(data_dir):
        if dir_ in skip:
            print(f"Skipped: Directory {dir_}")
            continue
        if dir_ not in label_map:
            label_map[dir_] = len(label_map)
        print(f"Processing: Directory {dir_} ")
        for img_path in os.listdir(os.path.join(data_dir, dir_)):
            data_aux = []

            x_ = []
            y_ = []
            z_ = []

            img = cv2.imread(os.path.join(data_dir, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
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

                if len(data_aux) != 63:
                    print(os.path.join(data_dir, dir_, img_path))
                data.append(data_aux)
                labels.append(label_map[dir_])

    # Convert data and labels to numpy arrays
    data = np.asarray(data)
    labels = np.asarray(labels)

    # Save data and labels as numpy arrays
    np.save(f"Data/numpy/{target}_data.npy", data)
    np.save(f"Data/numpy/{target}_labels.npy", labels)
    np.save(f"label_map/{lmap}_label_map.npy", label_map)


if __name__ == "__main__":
    # Preprocess Text data
    dataProcessor("Alphabet_Data/Train_Alphabet", "alphabets/train", "alphabet")
    dataProcessor("Alphabet_Data/Test_Alphabet", "alphabets/test", "alphabet")

    # Preprocess Numeric data
    dataProcessor("Numeric_Data/Train_Nums", "numbers/num_train", "num")
    dataProcessor("Numeric_Data/Test_Nums", "numbers/num_test", "num")
