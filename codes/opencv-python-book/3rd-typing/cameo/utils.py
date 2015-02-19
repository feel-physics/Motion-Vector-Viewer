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

def createCurveFunc(points):
    """
    制御点を元にした関数を返す
    xが入力、yが出力なので制御点(128, 160)はより明るくする
    :param points: 制御点
    :type  points: list[tuple]
    :return 補完された1次元の関数
    :rtype function
    """
    if points is None:
        return None
    numPoints = len(points)
    if numPoints < 2:
        return None
    xs, ys = zip(*points)
    # Works like itertools.izip().
    # TODO: 仮説(x1,y1),(x2,y2),(x3,y3)->(x1,x2,x3),(y1,y2,y3)
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

def createLookupArray(func, length = 256):
    """
    ピクセルごとにcreateCurveFuncしていたら大変なので
    1から256の入力に対する出力を変換用配列にしておく。
    :param func:   カーブ関数
    :param length: 入力の段階数
    :return: LookupArray
    """
    if func is None:
        return None
    lookupArray = numpy.empty(length)
    i = 0
    while i < length:
        func_i = func(i)
        func_i = max(0, func_i) # 出力値は0以上
        func_i = min(func_i, length - 1) # 出力値の最大はlength-1
        lookupArray[i] = func_i
        i += 1
    return lookupArray

def applyLookupArray(lookupArray, src, dst):
    """
    LookupArrayを使って入力画像から出力画像を求める
    :param lookupArray: あらかじめカーブ関数を変換用配列にしたもの
    :param src: グレースケールもしくはBGR形式の入力画像
    :param dst: グレースケールもしくはBGR形式の出力画像
    :return: None
    """
    if lookupArray is None:
        return
    dst[:] = lookupArray[src]
    # 左辺にスライスを付けているのはidを変えないため
    # スライスが付いていないと左辺のidが右辺のidによって上書きされてしまう。

def createCompositeFunc(func0, func1):
    """
    カーブ関数をあらかじめ合成する
    そうしておいてLookupArrayをつくる方が、
    何回もLookupArrayを適用するよりも効率的かつ正確になる
    :param func0: カーブ関数0
    :type  func0: function
    :param func1: カーブ関数1
    :type  func1: function
    :return: 合成されたカーブ関数
    :rtype : function
    """
    if func0 is None:
        return func1
    if func1 is None:
        return func0
    return lambda x: func0(func1(x))
    # >>> def make_incrementor(n):
    # ...     return lambda x: x + n
    # ...
    # >>> f = make_incrementor(42)
    # >>> f(0)
    # 42
    # >>> f(1)
    # 43
    #
    # >>> pairs = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
    # >>> pairs.sort(key=lambda pair: pair[1])
    # >>> pairs
    # [(4, 'four'), (1, 'one'), (3, 'three'), (2, 'two')]

# 何をしているのか不明。なくても動く。現在は使用してない。
def createFlatView(array):
    """
    入力された配列（何次元でも良い）の、1次元のビューを返す
    :param array: 配列
    :return: 1次元のビュー
    """
    flatView = array.view()
    # numpy.chararray.view
    # New view of array with the same data.
    # 同じデータの配列の新しいビュー
    # view()は型変換や要素型変換に使う
    flatView.shape = array.size
    return flatView