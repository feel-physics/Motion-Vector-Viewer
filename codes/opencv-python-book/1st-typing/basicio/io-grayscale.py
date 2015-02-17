__author__ = 'weed'

import cv2

grayImage = cv2.imread('hoge.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
cv2.imwrite('hoge-gray.png', grayImage)