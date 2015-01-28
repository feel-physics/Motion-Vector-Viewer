# coding=utf-8
__author__ = 'weed'

import cv2
import numpy
import os

# 120,000個のランダムなバイトの配列をつくる
randomByteArray = bytearray(os.urandom(120000))
""":type : bytearray"""
flatNumpyArray = numpy.array(randomByteArray)
""":type : numpy.ndarray"""

# 配列を変換して、400x300のグレースケール画像をつくる
grayImage = flatNumpyArray.reshape(300, 400)
""":type : numpy.ndarray"""
cv2.imwrite('RandomGray.png', grayImage)

# 配列を変換して、400x100のカラー画像をつくる
bgrImage = flatNumpyArray.reshape(100, 400, 3)
""":type : numpy.ndarray"""
cv2.imwrite('RandomColor.png', bgrImage)