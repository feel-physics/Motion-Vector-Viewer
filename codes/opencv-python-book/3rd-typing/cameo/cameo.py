# coding=utf-8
__author__ = 'weed'

import cv2
import filters
from managers import WindowManager, CaptureManager
from trackers import FaceTracker

class Cameo(object):

    def __init__(self):

        self._windowManager = WindowManager('Cameo',
                                            self.onKeypress)
        """:type : managers.WindowManager"""
        # def __init__(self, windowName, keypressCallback = None):

        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, True
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

            # 顔を検出して・・・
            self._faceTracker.update(frame)

            # これが何をやっているのかサッパリわからん
            # faces = self._faceTracker.faces
            # rects.swapRects(frame, frame,
            #                 [face.faceRect for face in faces])

            # 検出した領域の周りに枠を描画する
            if self._shouldDrawDebugRects:
                self._faceTracker.drawDebugRects(frame)

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
        :param keycode: intppp
        :return: None
        """
        if keycode == 32: # スペース
            self._captureManager.writeImage('screen-shot.png')
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

        elif keycode == ord('r'):
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

if __name__=="__main__":
    Cameo().run()