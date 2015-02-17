# coding=utf-8
__author__ = 'weed'

import cv2
import rects
import utils

class Face(object):
    """
    顔の特徴のデータ：顔、目、鼻、口
    """

    def __init__(self):
        """
        矩形のフォーマットは(x, y, w, h)
        左上の座標は(x, y)、右下の座標は(x+w, y+h)
        """
        self.faceRect     = None
        """:type : tuple"""
        self.leftEyeRect  = None
        """:type : tuple"""
        self.rightEyeRect = None
        """:type : tuple"""
        self.noseRect     = None
        """:type : tuple"""
        self.mouthRect    = None
        """:type : tuple"""

class FaceTracker(object):
    """
    顔の特徴（顔、目、鼻、口）を追跡するクラス
    """
    def __init__(self, scaleFactor = 1.2, minNeighbors = 2,
                 flags = cv2.cv.CV_HAAR_SCALE_IMAGE):

        self.scaleFactor  = scaleFactor
        """:type : float"""
        self.minNeighbors = minNeighbors
        """:type : int"""
        self.flags        = flags
        """:type : int"""

        self._faces = []
        """:type : list of [Face]"""

        self._faceClassifier = cv2.CascadeClassifier(
            'cascades/haarcascade_frontalface_alt.xml'
        )
        """:type : object"""
        self._eyeClassifier  = cv2.CascadeClassifier(
            'cascades/haarcascade_eye.xml'
        )
        """:type : object"""
        self._noseClassifier = cv2.CascadeClassifier(
            'cascades/haarcascade_mcs_nose.xml'
        )
        """:type : object"""
        self._mouthClassifier = cv2.CascadeClassifier(
            'cascades/haarcascade_mcs_mouth.xml'
        )

    @property
    def faces(self):
        """
        追跡する顔の特徴
        :return: list of [Face]
        """
        return self._faces

    def update(self, image):
        """
        追跡している顔の特徴を更新する
        :param image:
        :return:
        """

        self._faces = []

        if utils.isGray(image):
            image = cv2.equalizeHist(image)
        else:
            image = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
            cv2.equalizeHist(image, image)

        minSize = utils.withHeightDividedBy(image, 8)
        faceRects = self._faceClassifier.detectMultiScale(
            image, self.scaleFactor, self.minNeighbors, self.flags,
            minSize
        )

        if faceRects is not None:
            for faceRect in faceRects:

                face = Face()
                """:type : Face"""
                face.faceRect = faceRect

                x, y, w, h = faceRect

                # 顔の左上の部分から目を探す
                searchRect = (x+w/7, y, w*2/7, h/2)
                face.leftEyeRect = self._detectOneObject(
                    self._eyeClassifier, image, searchRect, 64
                )

                # 顔の右上の部分から目を探す
                searchRect = (x+w*4/7, y, w*2/7, h/2)
                face.rightEyeRect = self._detectOneObject(
                    self._eyeClassifier, image, searchRect, 64
                )

                # 顔の中央部分から鼻を探す
                searchRect = (x+w/4, y+h/4, w/2, h/2)
                face.noseRect = self._detectOneObject(
                    self._noseClassifier, image, searchRect, 32
                )

                # 顔の中央下寄りの部分から口を探す
                searchRect = (x+w/6, y+h*2/3, w*2/3, h/3)
                face.mouthRect = self._detectOneObject(
                    self._mouthClassifier, image, searchRect, 16
                )

                self._faces.append(face)

    def _detectOneObject(self, classifier, image, rect,
                         imageSizeToMinSizeRatio):
        """

        :param classifier:
        :param image:
        :param rect:
        :param imageSizeToMinSizeRatio:
        :return:
        """
        x, y, w, h = rect

        minSize = utils.withHeightDividedBy(
            image, imageSizeToMinSizeRatio
        )

        subImage = image[y:y+h, x:x+w]

        subRects = classifier.detectMultiScale(
            subImage, self.scaleFactor, self.minNeighbors, self.flags, minSize
        )

        if len(subRects) == 0:
            return None

        subX, subY, subW, subH = subRects[0]
        return (x+subX, y+subY, subW, subH)

    def drawDebugRects(self, image):
        """
        追跡している顔の特徴を囲む矩形を描画する
        :param image:
        :return:
        """
        if utils.isGray(image):
            faceColor     = 255
            leftEyeColor  = 255
            rightEyeColor = 255
            noseColor     = 255
            mouthColor    = 255
        else:
            faceColor     = (255, 255, 255) # 白
            leftEyeColor  = (  0,   0, 255) # 赤
            rightEyeColor = (  0, 255, 255) # 黄
            noseColor     = (  0, 255,   0) # 緑
            mouthColor    = (255,   0,   0) # 青

        for face in self.faces:
            rects.outlineRect(image, face.faceRect    , faceColor)
            rects.outlineRect(image, face.leftEyeRect , leftEyeColor)
            rects.outlineRect(image, face.rightEyeRect, rightEyeColor)
            rects.outlineRect(image, face.noseRect    , noseColor)
            rects.outlineRect(image, face.mouthRect   , mouthColor)