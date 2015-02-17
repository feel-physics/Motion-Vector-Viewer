# coding=utf-8
__author__ = 'weed'

import cv2
import numpy
import scipy.interpolate

def isGray(image):
    """
    画像がグレースケール画像ならTrueを返す
    :param image: 画像
    :type  image: numpy.ndarray
    :return: 真偽値
    """
    return image.ndim < 3
    # ndarray.ndim
    # Number of array dimensions.
    #
    # Examples
    #
    # >>>
    # >>> x = np.array([1, 2, 3])
    # >>> x.ndim
    # 1
    # >>> y = np.zeros((2, 3, 4))
    # >>> y.ndim
    # 3

def widthHeightDividedBy(image, divisor):
    """
    分割した画像の幅と高さを返す
    :param image: 画像
    :type  image: numpy.ndarray
    :param divisor: 分割する数
    :type  divisor: int
    :return: (分割された幅,分割された高さ)
    """
    h, w = image.shape[:2]
    # ndarray.shape
    # Tuple of array dimensions.
    #
    # Examples
    #
    # >>>
    # >>> x = np.array([1, 2, 3, 4])
    # >>> x.shape
    # (4,)
    # >>> y = np.zeros((2, 3, 4))
    # >>> y.shape
    # (2, 3, 4)
    # >>> y.shape = (3, 8)
    # >>> y
    # array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    # >>> y.shape = (3, 6)
    # Traceback (most recent call last):
    #   File "<stdin>", line 1, in <module>
    # ValueError: total size of new array must be unchanged

    return (w/divisor, h/divisor)

def createLookupArray(func, length = 256):
    """

    :param func:
    :param length:
    :return:
    """
    if func is None:
        return None
    lookupArray = numpy.empty(length)
    i = 0
    while i < length:
        func_i = func(i)
        lookupArray[i] = min(max(0, func_i), length - 1)
        i += 1
    return lookupArray

def createCurveFunc(points):
    """
    制御点を元にした関数を返す
    xが入力、yが出力なので制御点(128, 160)はより明るくする
    :param points: 制御点
    :return:
    """
    if points is None:
        return None
    numPoints = len(points)
    if numPoints < 2:
        return None
    xs, ys = zip(*points)
    # Works like itertools.izip().
    #
    ### TODO: これは何をやっているんだ？
    ### TODO: 仮説(x1,y1),(x2,y2)->(x1,x2),(y1,y2)
    #
    if numPoints < 4:
        kind = 'linear'
        # 'quadratic'（ベジェ曲線のようなもの） is not implemented
    else:
        kind = 'cubic' # ベジェ曲線
    return scipy.interpolate.interp1d(xs, ys, kind,
                                      bounds_error=False)
    # class scipy.interpolate.interp1d(x, y, kind='linear')
    # Interpolate a 1-D function.
    #
    # x and y are arrays of values used to approximate some function f: y = f(x).
    # This class returns a function whose call method uses interpolation to find the value of new points.
