# coding=utf-8
__author__ = 'weed'

import cv2
cap = cv2.VideoCapture(0)


# Video Record Code
import lib_video
videoWriter = lib_video.initVideoRecoder(cap)


fgbg = cv2.BackgroundSubtractorMOG2()
while True:
    ret, src = cap.read()
    fgmask = fgbg.apply(src, learningRate=0.01)
    dst = src.copy()
    dst = cv2.bitwise_and(src, src, mask=fgmask)
    cv2.imshow('frame',dst)


    videoWriter.write(dst)  # Video Record Code


    k = cv2.waitKey(30) & 0xff
    if k == 27:  # ESC key
        break
cap.release()
cv2.destroyAllWindows()