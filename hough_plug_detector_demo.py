import sys, os
import numpy as np
# import matplotlib.pyplot as plt
import cv2

webcam = cv2.VideoCapture(1)
size = 2

while True:
    rval, color = webcam.read()
    while not rval:
        rval, color = webcam.read()
        if not rval:
            print("Trying again...")

    #color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    mini = cv2.resize(frame, (int(frame.shape[1] / size), int(frame.shape[0] / size)))
    img = cv2.medianBlur(mini, 5)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=120, param2=60)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(circles[0, :])
        for i in circles[0, :]:
            cv2.circle(color, (i[0] * 2, i[1] * 2), i[2] * 2, (0, 255, 0), 2)
            cv2.circle(color, (i[0] * 2, i[1] * 2), 2, (0, 0, 255), 3)


    cv2.imshow('Detector', color)
    key = cv2.waitKey(500)
    #spacebar
    if key == 32:
        break
