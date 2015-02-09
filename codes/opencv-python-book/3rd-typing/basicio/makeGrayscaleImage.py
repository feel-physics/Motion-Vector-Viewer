# coding=utf-8
__author__ = 'weed'

import cv2

grayImage = cv2.imread('screen-shot.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
""":type numpy.ndarray"""
cv2.imwrite('screen-shot-gray.png', grayImage)