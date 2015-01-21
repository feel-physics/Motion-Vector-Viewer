# coding=utf-8
__author__ = 'weed'

import cv2
import numpy
import os

# 120000個の乱数配列を作る
randomByteArray = bytearray(os.urandom(120000))
flatNumpyArray = numpy.array(randomByteArray)

# 配列を、400x300のグレースケールの画像に変換する
grayImage = flatNumpyArray.reshape(300, 400)
cv2.imwrite('RandomGray.png', grayImage)

# 配列を、400x100のカラー画像に変換する
bgrImage = flatNumpyArray.reshape(100, 400, 3)
cv2.imwrite('RandomColor.png', bgrImage)