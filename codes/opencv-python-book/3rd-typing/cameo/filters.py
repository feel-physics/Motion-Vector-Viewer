# coding=utf-8
__author__ = 'weed'

import cv2
import numpy
import utils

class BGRFuncFilter(object):
    """

    """

    def __init__(self, vFunc = None, bFunc = None, gFunc = None,
                 rFunc = None, dtype = numpy.unit8):
        lengh = numpy.iinfo(dtype).max + 1
        self._vLookupArray = utils.createLookupArray(vFunc, lengh)

    def apply(self, src, dst):
        """

        :param src:
        :param dst:
        :return:
        """


class BGRPortalCurveFilter