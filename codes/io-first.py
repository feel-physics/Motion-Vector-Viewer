__author__ = 'weed'

import cv2

image = cv2.imread('hoge.png')
cv2.imwrite('hoge.jpg', image)