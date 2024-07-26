import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2 as cv
from HandTracking import HandDetector
import numpy as np
import math
import time

cap = cv.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 30
frameSize = 300

counter = 0
current_dir = os.path.dirname(__file__)

time.sleep(2)
dir_name = input("Enter Directory Name: ")  # eg., A,B,C,0,1 etc.

parent = "Data/Sample"  # Change the folder as per your choice

if not os.path.exists(parent):
    os.mkdir(parent)
folder = os.path.join(current_dir, f"Data/Sample/{dir_name}")


if not os.path.exists(folder):
    os.mkdir(folder)

while True:
    _, frame = cap.read()

    try:
        hands, frame = detector.findHands(frame, draw=False)
        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]

            imgWhite = np.ones((frameSize, frameSize, 3), np.uint8) * 255
            frameCrop = frame[y - offset : y + h + offset, x - offset : x + w + offset]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = frameSize / h
                wCal = math.ceil(k * w)
                wGap = math.ceil((frameSize - wCal) / 2)
                frameResize = cv.resize(frameCrop, (wCal, frameSize))
                frameResizeShape = frameResize.shape
                imgWhite[:, wGap : wCal + wGap] = frameResize

            else:
                k = frameSize / w
                hCal = math.ceil(k * h)
                hGap = math.ceil((frameSize - hCal) / 2)

                frameResize = cv.resize(frameCrop, (frameSize, hCal))
                frameResizeShape = frameResize.shape
                imgWhite[hGap : hCal + hGap, :] = frameResize

            cv.imshow("Cropped", frameCrop)
            cv.imshow("White Frame", imgWhite)
        cv.imshow("Image", frame)
        key = cv.waitKey(1)

        # Press 's' to save image
        if key == ord("s"):
            counter += 1
            cv.imwrite(f"{folder}/image_{time.time()}.jpg", imgWhite)
            print(f"{counter} Success!")

        # Press 'q' to quit
        elif key == ord("q"):
            print("Closing...")
            break

    except Exception as e:
        print(f"An error occured: {e}")
