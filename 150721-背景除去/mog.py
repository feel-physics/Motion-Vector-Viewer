# coding=utf-8
__author__ = 'weed'

import cv2
import lib_video

cap = cv2.VideoCapture(0)


videoWriter = lib_video.initVideoRecoder(cap)  # Video Record Code


fgbg = cv2.BackgroundSubtractorMOG()
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