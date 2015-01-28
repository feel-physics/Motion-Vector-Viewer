# coding=utf-8
__author__ = 'weed'

import cv2

videoCapture = cv2.VideoCapture('sample.avi')
fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
size = (
    int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH )),
    int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
)
videoWriter = cv2.VideoWriter(
    'output.avi', cv2.cv.CV_FOURCC('I','4','2','0'), fps, size
)
success, frame = videoCapture.read()
while success:
    videoWriter.write(frame)
    success, frame = videoCapture.read()