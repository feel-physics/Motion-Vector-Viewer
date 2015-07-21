# coding=utf-8
__author__ = 'weed'

import cv2
from datetime import datetime

cap = cv2.VideoCapture(0)


fps = 30
size = (
    int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH )),
    int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
)
file_name = datetime.now().strftime('%y%m%d-%H%M%S') + '-screencast.avi'
videoWriter = cv2.VideoWriter(
    file_name,
    cv2.cv.CV_FOURCC('I', '4', '2', '0'),  # aviファイル形式
    fps,
    size)


fgbg = cv2.BackgroundSubtractorMOG()
while True:
    ret, src = cap.read()
    fgmask = fgbg.apply(src, learningRate=0.01)
    dst = src.copy()
    dst = cv2.bitwise_and(src, src, mask=fgmask)
    cv2.imshow('frame',dst)


    videoWriter.write(dst)


    k = cv2.waitKey(30) & 0xff
    if k == 27:  # ESC key
        break
cap.release()
cv2.destroyAllWindows()