# coding=utf-8
__author__ = 'weed'

import cv2
import numpy
import utils

class BGRFuncFilter(object):
    """

    """

    def __init__(self, vFunc = None, bFunc = None, gFunc = None,
                 rFunc = None, dType = numpy.uint8):
        """

        :param vFunc:
        :param bFunc:
        :type  bFunc: function
        :param gFunc:
        :param rFunc:
        :param dType:
        :return:
        """
        lengh = numpy.iinfo(dType).max + 1
        self._bLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(bFunc, vFunc), lengh)
        self._gLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(gFunc, vFunc), lengh)
        self._rLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(rFunc, vFunc), lengh)

    def apply(self, src, dst):
        """

        :param src:
        :param dst:
        :return:
        """
        b, g, r = cv2.split(src)
        utils.applyLookupArray(self._bLookupArray, b, b)
        utils.applyLookupArray(self._gLookupArray, g, g)
        utils.applyLookupArray(self._rLookupArray, r, r)
        cv2.merge([b, g, r], dst)

class BGRCurveFilter(BGRFuncFilter):
    """

    """
    def __init__(self, vPoints = None, bPoints = None,
                 gPoints = None, rPoints = None,
                 dType = numpy.uint8):
        BGRFuncFilter.__init__(self,
                               utils.createCurveFunc(vPoints),
                               utils.createCurveFunc(bPoints),
                               utils.createCurveFunc(gPoints),
                               dType)

class BGRPortraCurveFilter(BGRCurveFilter):
    """

    """
    def __init__(self, dType = numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            bPoints=[(0,0),(35,25),(205,227),(255,255)],
            gPoints=[(0,0),(27,21),(196,207),(255,255)],
            rPoints=[(0,0),(59,54),(202,210),(255,255)],
            dType=dType
        )

def recolorRC(src, dst):
    """
    BGRからRC（レッド、シアン）に変換する
    :param src:
    :param dst:
    :return:
    """
    b, g, r = cv2.split(src)
    # Python: cv2.split(m[, mv]) → mv
    # Parameters:
    # src – input multi-channel array.
    # mv – output array or vector of arrays; in the first variant of the function the number of arrays must match src.channels(); the arrays themselves are reallocated, if needed.
    # The functions split split a multi-channel array into separate single-channel arrays
    """:type b : numpy.ndarray"""
    """:type g : numpy.ndarray"""
    """:type r : numpy.ndarray"""
    cv2.addWeighted(b, 0.5, g, 0.5, 0, b)
    cv2.merge((b, b, r), dst)

def recolorRGV(src, dst):
    """
    BGRからRGV（赤、緑、値）
    :param src:
    :param dst:
    :return:
    """
    b, g, r = cv2.split(src)
    cv2.min(b, g, b)
    cv2.min(b, r, b)
    cv2.merge((b, g, r), dst)

def recolorCMV(src, dst):
    """
    BGRからCMV（シアン、マゼンタ、値）に変換する
    :param src:
    :param dst:
    :return:
    """
    b, g, r = cv2.split(src)
    cv2.max(b, g, b)
    cv2.max(b, r, b)
    cv2.merge((b, g, r), dst)

def strokeEdges(src, dst, blurKsize = 7, edgeKsize = 5):
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.cv.CV_8U, graySrc, ksize=edgeKsize)
    # Python: cv2.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]) → dst
    # C: void cvLaplace(const CvArr* src, CvArr* dst, int aperture_size=3 )
    # Python: cv.Laplace(src, dst, apertureSize=3) → None
    # Parameters:
    # src – Source image.
    # dst – Destination image of the same size and the same number of channels as src .
    # ddepth – Desired depth of the destination image.
    # ksize – Aperture size used to compute the second-derivative filters. See getDerivKernels() for details. The size must be positive and odd.
    # scale – Optional scale factor for the computed Laplacian values. By default, no scaling is applied. See getDerivKernels() for details.
    # delta – Optional delta value that is added to the results prior to storing them in dst .
    # borderType – Pixel extrapolation method. See borderInterpolate() for details.
    # The function calculates the Laplacian of the source image by adding up the second x and y derivatives calculated using the Sobel operator:
    #
    # \texttt{dst} =  \Delta \texttt{src} =  \frac{\partial^2 \texttt{src}}{\partial x^2} +  \frac{\partial^2 \texttt{src}}{\partial y^2}
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)

class VConvolutionFilter(object):
    """

    """
    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        """

        :param src:
        :param dst:
        :return:
        """
        cv2.filter2D(src, -1, self._kernel, dst)

class SharpenFilter(VConvolutionFilter):
    """

    """
    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)

class FindEdgesFilter(VConvolutionFilter):
    """

    """
    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)

class BlurFilter(VConvolutionFilter):
    """
    半径2ピクセルでぼかす
    """
    def __init__(self):
        kernel = numpy.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)

class EmbossFilter(VConvolutionFilter):
    """

    """
    def __init__(self):
        kernel = numpy.array([[-2, -1,  0],
                              [-1,  1,  1],
                              [ 0,  1,  2]])
        VConvolutionFilter.__init__(self, kernel)

def convertBgr2Hsv(src, dst):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(src)
    s[:,:] = 0
    cv2.merge((h, s, v), src)
    cv2.cvtColor(src, cv2.COLOR_HSV2BGR, dst)