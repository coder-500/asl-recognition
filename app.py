import cv2
import datetime
from gtts import gTTS
import mediapipe as mp
import numpy as np
import os
from playsound import playsound
import streamlit as st
from tensorflow.keras.models import load_model
import time
import uuid


def text_detection_model_load():
    # Load the model and label map
    model = load_model("models/asl_alphabet_model.h5")
    label_map = np.load("label_map/alphabet_label_map.npy", allow_pickle=True).item()

    # Reverse label map for prediction
    reverse_label_map = {v: k for k, v in label_map.items()}

    return model, reverse_label_map


def num_detection_model_load():
    # Load the model and label map
    model = load_model("models/asl_number_model.h5")
    label_map = np.load("label_map/num_label_map.npy", allow_pickle=True).item()

    # Reverse label map for prediction
    reverse_label_map = {v: k for k, v in label_map.items()}

    return model, reverse_label_map


def detection(model, reverse_label_map):
    if st.session_state.run:
        # Open webcam
        cap = cv2.VideoCapture(0)

        predicted_class = ""
        prev_pred = ""

        word = ""
        word_list = []
        sentence = ""

        count_same_frame = 0
        start = time.time()

        while True:
            try:
                data_aux = []
                x_ = []
                y_ = []
                z_ = []
                ret, frame = cap.read()

                if not ret:
                    st.warning(
                        " Failed to connect to webcam! Please check the connection and try again! ⚠️"
                    )
                    break

                h, w, _ = frame.shape

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    for lm in results.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame,
                            lm,
                            mp_hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style(),
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

                    prev_pred = predicted_class

                    data_aux = np.array(data_aux).reshape(1, 21, 3, 1)
                    prediction = model.predict(data_aux)
                    predicted_class = reverse_label_map[np.argmax(prediction)]

                    if prev_pred == predicted_class:
                        count_same_frame += 1
                    else:
                        count_same_frame = 0

                    if count_same_frame > 8:
                        word += predicted_class
                        word_list.append(predicted_class)
                        word_placeholder.markdown(
                            f"<p>Output: <span class='out'>{word}</span></p>",
                            unsafe_allow_html=True,
                        )
                        count_same_frame = 0

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
                    placeholder.markdown(
                        f"<p>Prediction: <span class='out'>{predicted_class}</span></p>",
                        unsafe_allow_html=True,
                    )
                    start = time.time()
                else:
                    end = time.time() - start
                    placeholder.markdown(
                        f"<p>Prediction: </p>",
                        unsafe_allow_html=True,
                    )
                    word += " "
                    if end > 15:
                        word = " "
                        sentence = ""
                    word_placeholder.markdown(
                        f"<p>Output: <span class='out'>{word}</span></p>",
                        unsafe_allow_html=True,
                    )

                st_frame.image(frame, caption="Live Feed", channels="BGR")

                # Text to speech
                if not results.multi_hand_landmarks and word_list:
                    try:
                        words = "".join(word_list).lower()
                        sentence += words + " "
                        if words == "ooooo" or words == "10101":
                            if sentence:
                                sentence = sentence[:-6]
                                text_to_speech_gtts(sentence)
                                word = word[:-6]
                            else:
                                pass
                        else:
                            text_to_speech_gtts(words)
                        word_list = []
                    except Exception as e:
                        print(f"Error: {e}")

            except Exception as e:
                print(f"An Error occurred: {e}")

        cap.release()
        cv2.destroyAllWindows()

    else:
        # Placeholder image when webcam is off
        st_frame.image(np.ones((480, 640, 3)), channels="BGR", caption="Enable Camera")
        placeholder.markdown(
            f"<p>Prediction: </p>",
            unsafe_allow_html=True,
        )
        word_placeholder.markdown(
            f"<p>Output: </p>",
            unsafe_allow_html=True,
        )


def start_stop():
    st.session_state.run = not st.session_state.run


def clear_txt():
    word_placeholder.markdown("<p></p>", unsafe_allow_html=True)


def text_to_speech_gtts(txt):
    unique_file = f"assets/voice_{uuid.uuid4()}.mp3"
    lang = "en"
    obj = gTTS(text=txt, lang=lang, slow=False)
    obj.save(unique_file)
    playsound(unique_file)
    os.remove(unique_file)


if __name__ == "__main__":

    st.image("./assets/asl_2.png")
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Initialize Mediapipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    st.title("ASL Recognition System ...")

    # Access webcam and perform inference
    if "run" not in st.session_state:
        st.session_state.run = False

    # Align buttons
    col1, col2, col3, _ = st.columns([1, 1, 1, 1])

    with st.container(border=True):
        with col1:
            if st.session_state.run:
                start_button = st.button("Disable Camera", on_click=start_stop)
            else:
                start_button = st.button("Enable Camera", on_click=start_stop)
        with col2:
            clear_button = st.button("Clear Output", on_click=clear_txt)

    # Empty placeholders
    with st.container(border=True):
        st_frame = st.empty()
        placeholder = st.empty()
        word_placeholder = st.empty()

    with st.sidebar:
        choose_model = st.radio(
            "Choose model type", ("ASL Text Detection Model", "ASL Number Detection")
        )
        year = datetime.datetime.today().year
        st.markdown(
            f'<p class="footerText">&copy {year} | All Rights Reserved</p>',
            unsafe_allow_html=True,
        )

    if choose_model == "ASL Text Detection Model":
        model, reverse_label_map = text_detection_model_load()

    elif choose_model == "ASL Number Detection":
        model, reverse_label_map = num_detection_model_load()

    detection(model, reverse_label_map)

    with st.expander("How to use"):
        st.write(
            """
            Direction of use...
        """
        )
        st.image("https://static.streamlit.io/examples/dice.jpg")
