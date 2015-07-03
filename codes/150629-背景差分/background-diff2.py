# coding=utf-8
__author__ = 'weed'

import cv2

cam = cv2.VideoCapture(0)
winName = "Movement Indicator"
cv2.namedWindow(winName, cv2.CV_WINDOW_AUTOSIZE)

img_past = None
img_now  = None

while True:
    img_past = img_now
    img_now  = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
    if img_now is not None and img_past is not None:
        img_diff = cv2.absdiff(img_now, img_past)
        cv2.imshow(winName, img_diff)
    key = cv2.waitKey(10)

print "Goodbye"