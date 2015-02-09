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