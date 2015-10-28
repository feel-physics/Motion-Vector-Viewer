# coding=utf-8
__author__ = 'weed'

import cv2
from datetime import datetime

def initVideoRecoder(cap):
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
    return videoWriter