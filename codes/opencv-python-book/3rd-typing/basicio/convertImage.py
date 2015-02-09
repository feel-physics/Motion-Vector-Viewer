# coding=utf-8
__author__ = 'weed'

import cv2

image = cv2.imread('screen-shot.png')
""":type : numpy.ndarray"""
cv2.imwrite('screen-shot.jpg', image)
