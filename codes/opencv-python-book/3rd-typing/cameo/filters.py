# coding=utf-8
__author__ = 'weed'

import cv2
import numpy
import utils

def recolorRC(src, dst):
    """
    BGRからRC（赤、シアン）への変換をシミュレーションする
    コードの内容：
    dst.b = dst.g = 0.5 * (src.b + src.g)
    dst.r = src.r
    :param src: BGR形式の入力画像
    :param dst: BGR形式の出力画像
    :return: None
    """
    b, g, r = cv2.split(src)
    cv2.addWeighted(b, 0.5, g, 0.5, 0, b)
    # Python: cv.AddWeighted(src1, alpha, src2, beta, gamma, dst) → None
    # Parameters:
    # src1 – first input array.
    # alpha – weight of the first array elements.
    # src2 – second input array of the same size and channel number as src1.
    # beta – weight of the second array elements.
    # dst – output array that has the same size and number of channels as the input arrays.
    # gamma – scalar added to each sum.
    # dtype – optional depth of the output array; when both input arrays have the same depth, dtype can be set to -1, which will be equivalent to src1.depth().
    # The function addWeighted calculates the weighted sum of two arrays as follows:
    # dst = src1 * alpha + src2 * beta + gamma
    cv2.merge((b, b, r), dst)

def recolorRGV(src, dst):
    """
    BGRからRGV（赤、緑、値）への変換をシミュレートする
    コードの内容：
    dst.b = min(src.b, src.g, src.r)
    dst.g = src.g
    dst.r = src.r
    :param src: BGR形式の入力画像
    :param dst: BGR形式の出力画像
    :return: None
    """
    b, g, r = cv2.split(src)
    cv2.min(b, g, b)
    cv2.min(b, r, b)
    # Python: cv2.min(src1, src2[, dst]) → dst
    # Python: cv.Min(src1, src2, dst) → None
    # Python: cv.MinS(src, value, dst) → None
    # Parameters:
    # src1 – first input array.
    # src2 – second input array of the same size and type as src1.
    # value – real scalar value.
    # dst – output array of the same size and type as src1.
    # The functions min calculate the per-element minimum of two arrays:
    #     dst = min(src1, src2)
    # or array and a scalar:
    #     dst = min(src1, value)
    # In the second variant, when the input array is multi-channel, each channel is compared with value independently.
    cv2.merge((b, g, r), dst)

def recolorCMV(src, dst):
    """
    BGRからCMV（シアン、マゼンタ、値）への変換をシミュレートする
    コードの内容：
    dst.b = max(src.b, src.g, src.r)
    dst.g = src.g
    dst.r = src.r
    :param src: BGR形式の入力画像
    :param dst: BGR形式の出力画像
    :return: None
    """
    b, g, r = cv2.split(src)
    cv2.max(b, g, b)
    cv2.max(b, r, b)
    cv2.merge((b, g, r), dst)