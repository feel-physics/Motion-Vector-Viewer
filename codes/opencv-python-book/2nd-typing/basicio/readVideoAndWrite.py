# coding=utf-8
__author__ = 'weed'

import cv2
import time

videoCapture = cv2.VideoCapture('sample.avi')
print(type(videoCapture))
fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
size = (
    int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH )),
    int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
)
encoding = cv2.cv.CV_FOURCC('I','4','2','0')
print(type(encoding))
videoWriter = cv2.VideoWriter(
    'output.avi',encoding , fps, size
)

startTime = time.time()
print(type(startTime))

framesElapsed = long(0)
print(type(framesElapsed))

fpsEstimate = framesElapsed / startTime
print(type(fpsEstimate))

success, frame = videoCapture.read()
print(type(frame))
while success:
    videoWriter.write(frame)
    success, frame = videoCapture.read()

keycode = cv2.waitKey(1)
print(type(keycode))