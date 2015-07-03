# coding=utf-8
__author__ = 'weed'

import cv2, time, numpy
from datetime import datetime

thresh_diff = 20
timeArrayToCalcFps = []

cam = cv2.VideoCapture(0)
cv2.namedWindow('Window', cv2.CV_WINDOW_AUTOSIZE)

time.sleep(1)
img_back = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)

while True:
    img_now  = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
    img_diff = cv2.absdiff(img_back, img_now)
    # 2値化 10以上離れていたら255
    _, img_mask = cv2.threshold(img_diff, thresh_diff, 255, cv2.THRESH_BINARY)

    # img_mask はノイジーなのでそれを除去する
    # 8近傍
    element8 = numpy.array([[1,1,1],
                            [1,1,1],
                            [1,1,1]], numpy.uint8)
    # オープニング
    cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, element8, img_mask, None, 2)

    img_show = cv2.bitwise_and(img_now, img_mask)
    cv2.imshow('Window', img_show)

    # FPSを計算する
    # 開始直後は・・・
    if len(timeArrayToCalcFps) < 10:
        # 日時を配列に追加していく
        timeArrayToCalcFps.append(datetime.now())
        fps = -1
    # あるていど日時がたまったら・・・
    else:
        # 相変わらず日時を配列に追加していくが、
        timeArrayToCalcFps.append(datetime.now())
        # 古い日時を棄て、
        timeArrayToCalcFps.pop(0)
        # 経過時間を求め、FPSを求める
        timeElapsed = timeArrayToCalcFps[9] - timeArrayToCalcFps[0]
        fps = 10 / (timeElapsed.seconds + timeElapsed.microseconds / 1000000.0)
        print "{0:.1f}".format(fps)

    keycode = cv2.waitKey(10)
    # GTKによってエンコードされた非ASCII情報を捨てる
    keycode &= 0xFF

    if keycode == 0 or keycode == 1:  # up / down arrow
        pitch = 10 if keycode == 1 else -10
        thresh_diff += pitch
        print thresh_diff