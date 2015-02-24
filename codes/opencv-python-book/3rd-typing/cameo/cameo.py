# coding=utf-8
__author__ = 'weed'

import cv2
import filters
from managers import WindowManager, CaptureManager
from trackers import FaceTracker
from datetime import datetime
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class Cameo(object):

    def __init__(self):

        self._windowManager = WindowManager('Cameo',
                                            self.onKeypress)
        """:type : managers.WindowManager"""
        # def __init__(self, windowName, keypressCallback = None):

        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, False
        )
        """:type : managers.CaptureManager"""
        # def __init__(self, capture, previewWindowManager = None,
        #     shouldMirrorPreview = False):

        self._curveFilter = filters.BGRPortraCurveFilter()
        self._testCurveFilter = filters.TestCurveFilter()

        self._faceTracker = FaceTracker()

        ### Filters ###
        self._shouldApplyPortraCurveFilter = False
        self._shouldStrokeEdge             = False
        self._shouldApplyBlur              = False
        self._shouldApplyLaplacian         = False
        self._shouldApplyTestCurveFilter   = False
        self._shouldRecolorRC = False
        self._shouldRecolorRGV = False
        self._shouldRecolorCMV = False
        self._shouldHueMask = False
        self._hue = 180

        self._shouldDrawDebugRects = False

    def run(self):
        """
        メインループを実行する
        :return:
        """
        # ウィンドウをつくる
        self._windowManager.createWindow()
        # ウィンドウが存在する限り・・・
        while self._windowManager.isWindowCreated:
            # フレームを取得し・・・
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            """:type : numpy.ndarray"""

            logging.debug('ca.frame(B)   : ' + str(id(frame)))

            ### Filters ###
            if self._shouldApplyPortraCurveFilter:
                self._curveFilter.apply(frame, frame)
            if self._shouldStrokeEdge:
                filters.strokeEdges(frame, frame)
            if self._shouldApplyBlur:
                filters.applyBlur(frame, frame)
            if self._shouldApplyLaplacian:
                filters.applyLaplacian(frame, frame)
            if self._shouldApplyTestCurveFilter:
                self._testCurveFilter.apply(frame, frame)
            if self._shouldRecolorRC:
                filters.recolorRC(frame, frame)
            if self._shouldRecolorRGV:
                filters.recolorRGV(frame, frame)
            if self._shouldRecolorCMV:
                filters.recolorCMV(frame, frame)
            if self._shouldHueMask:
                filters.hueMask(frame, frame, self._hue)

            # 顔を検出して・・・
            self._faceTracker.update(frame)

            # これが何をやっているのかサッパリわからん
            # faces = self._faceTracker.faces
            # rects.swapRects(frame, frame,
            #                 [face.faceRect for face in faces])

            # 検出した領域の周りに枠を描画する
            if self._shouldDrawDebugRects:
                self._faceTracker.drawDebugRects(frame)

            logging.debug('ca.frame(A)   : ' + str(id(frame)))

            # フレームを解放する
            self._captureManager.exitFrame()
            # キーイベントがあれば実行する
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """
        キー入力を処理するe
        スペース　：スクリーンショットを撮る
        タブ　　　：スクリーンキャストの録画を開始／終了する
        エスケープ：終了する
        :param keycode: int
        :return: None
        """
        if keycode == 32: # スペース
            self._captureManager.paused = \
                not self._captureManager.paused
        elif keycode == 13: # リターン
            now = datetime.now()
            timestamp = now.strftime('%y%m%d-%H%M%S')
            self._captureManager.writeImage(timestamp + '-screen-shot.png')

        elif keycode == 9: # タブ
            # 動画ファイルに書き出し中でなければ・・・
            if not self._captureManager.isWritingVideo:
                # ファイルに書き出すのを始めて・・・
                self._captureManager.startWritingVideo('screen-cast.avi')
            # 書き出し中であれば・・・
            else:
                # ・・・書き出しを終える
                self._captureManager.stopWritingVideo()
        elif keycode == 27: # エスケープ
            self._windowManager.destroyWindow()

        elif keycode == ord('d'):
            self._shouldDrawDebugRects = \
                not self._shouldDrawDebugRects

        ### Filters ###
        elif keycode == ord('p'):
            self._shouldApplyPortraCurveFilter = \
                not self._shouldApplyPortraCurveFilter
        elif keycode == ord('e'):
            self._shouldStrokeEdge = \
                not self._shouldStrokeEdge
        elif keycode == ord('b'):
            self._shouldApplyBlur = \
                not self._shouldApplyBlur
        elif keycode == ord('l'):
            self._shouldApplyLaplacian = \
                not self._shouldApplyLaplacian
        elif keycode == ord('t'):
            self._shouldApplyTestCurveFilter = \
                not self._shouldApplyTestCurveFilter

        ### Filters from 2nd typing ###
        elif keycode == ord('r'):
            self._shouldRecolorRC = \
                not self._shouldRecolorRC
        elif keycode == ord('v'):
            self._shouldRecolorRGV = \
                not self._shouldRecolorRGV
        elif keycode == ord('m'):
            self._shouldRecolorCMV = \
                not self._shouldRecolorCMV

        ### Hue Filter ###
        elif keycode == ord('h'):
            self._shouldHueMask = \
                not self._shouldHueMask
        elif keycode == 0: # up arrow
            self._hue += 10
            print self._hue
        elif keycode == 1: # down arrow
            self._hue -= 10
            print self._hue

        else:
            print keycode

if __name__=="__main__":
    Cameo().run()