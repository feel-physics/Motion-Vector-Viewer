# coding=utf-8
__author__ = 'weed'

import cv2
import numpy
import scipy.interpolate

def isGray(image):
    """
    グレースケール画像であるか返す
    具体的には、画像がピクセルごとに1チャンネルしか持たない場合にTrueを返す
    :param image: ndarray
    :return: boolean
    """
    return image.ndim < 3

def withHeightDividedBy(image, divisor):
    """
    変数で割った画像の次元を返す
    :param image: ndarray
    :param divisor: object
    :return: object
    """
    h, w = image.shape[:2]
    return (w/divisor, h/divisor)