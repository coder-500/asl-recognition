import av
import cv2
import datetime
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading


# @st.cache_resource(max_entries=1000, show_spinner=False)
# def text_detection_model_load():
#     # Load the model and label map
#     model = load_model("models/asl_alphabet_model.h5")
#     label_map = np.load("label_map/alphabet_label_map.npy", allow_pickle=True).item()

#     # Reverse label map for prediction
#     reverse_label_map = {v: k for k, v in label_map.items()}

#     return model, reverse_label_map


# @st.cache_resource(max_entries=1000, show_spinner=False)
# def num_detection_model_load():
#     # Load the model and label map
#     model = load_model("models/asl_number_model.h5")
#     label_map = np.load("label_map/num_label_map.npy", allow_pickle=True).item()

#     # Reverse label map for prediction
#     reverse_label_map = {v: k for k, v in label_map.items()}

#     return model, reverse_label_map


# 061624
@st.cache_resource(max_entries=1000, show_spinner=False)
def text_detection_model_load():
    interpreter = tf.lite.Interpreter(model_path="models/asl_alphabet_model.tflite")
    interpreter.allocate_tensors()
    label_map = np.load("label_map/alphabet_label_map.npy", allow_pickle=True).item()
    reverse_label_map = {v: k for k, v in label_map.items()}
    return interpreter, reverse_label_map


@st.cache_resource(max_entries=1000, show_spinner=False)
def num_detection_model_load():
    interpreter = tf.lite.Interpreter(model_path="models/asl_number_model.tflite")
    interpreter.allocate_tensors()
    label_map = np.load("label_map/num_label_map.npy", allow_pickle=True).item()
    reverse_label_map = {v: k for k, v in label_map.items()}
    return interpreter, reverse_label_map


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    if flag:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    with lock:
        frame_container["frame"] = img

    del frame  # 061524
    return av.VideoFrame.from_ndarray(img, format="bgr24")


def detect_with_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = input_data.astype(np.float32)  # 061724

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    return output_data


def detection():
    predicted_class = ""
    prev_pred = ""

    word = ""

    count_same_frame = 0
    start = time.time()

    global x1, y1, x2, y2, flag

    while ctx.state.playing:
        try:
            with lock:
                frame = frame_container["frame"]

            if frame is None:
                continue

            data_aux = []
            x_ = []
            y_ = []
            z_ = []

            h, w, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                flag = True
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

                x1 = int(min(x_) * w) - 25
                y1 = int(min(y_) * h) - 25

                x2 = int(max(x_) * w) + 25
                y2 = int(max(y_) * h) + 25

                prev_pred = predicted_class

                data_aux = np.array(data_aux).reshape(1, 21, 3, 1)
                # prediction = model.predict(data_aux)
                prediction = detect_with_tflite(model, data_aux)  # 061624
                predicted_class = reverse_label_map[np.argmax(prediction)]

                if prev_pred == predicted_class:
                    count_same_frame += 1
                else:
                    count_same_frame = 0

                # if count_same_frame > 7:
                if count_same_frame > 10:  # 061724
                    word += predicted_class
                    word_placeholder.markdown(
                        f"<p>Output: <span class='out'>{word}</span></p>",
                        unsafe_allow_html=True,
                    )
                    count_same_frame = 0

                placeholder.markdown(
                    f"<p>Prediction: <span class='out'>{predicted_class}</span></p>",
                    unsafe_allow_html=True,
                )
                start = time.time()
            else:
                flag = False
                end = time.time() - start
                placeholder.markdown(
                    f"<p>Prediction: </p>",
                    unsafe_allow_html=True,
                )
                word += " "
                if end > 15:
                    word = " "
                word_placeholder.markdown(
                    f"<p>Output: <span class='out'>{word}</span></p>",
                    unsafe_allow_html=True,
                )

            # del frame, frame_rgb, results, data_aux, x_, y_, z_  # 061524
        except Exception as e:
            print(f"An Error occurred: {e}")
    else:
        placeholder.markdown(
            f"<p>Prediction: </p>",
            unsafe_allow_html=True,
        )
        word_placeholder.markdown(
            f"<p>Output: </p>",
            unsafe_allow_html=True,
        )


def clear_txt():
    word_placeholder.markdown("<p></p>", unsafe_allow_html=True)


if __name__ == "__main__":

    lock = threading.Lock()
    frame_container = {"frame": None}

    x1 = None
    y1 = None
    x2 = None
    y2 = None
    flag = False

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

    clear_button = st.button("Clear Output", on_click=clear_txt)

    ctx = webrtc_streamer(
        key="test",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

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

    detection()

    with st.expander("How to use"):
        st.write(
            """
            Direction of use...
        """
        )
        st.image("https://static.streamlit.io/examples/dice.jpg")
