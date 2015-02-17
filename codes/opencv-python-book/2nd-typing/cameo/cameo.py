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
        # def __init__(self, capture, previewWindowManager = None,d
        #     shouldMirrorPreview = False):

        self._curveFilter = filters.BGRPortraCurveFilter()
        self._sharpenFilter = filters.SharpenFilter()
        self._findEdgesFilter = filters.FindEdgesFilter()
        self._blurFilter = filters.BlurFilter()
        self._embossFilter = filters.EmbossFilter()

        self._faceTracker = FaceTracker()

        self._shouldDrawDebugRects = False
        self._shouldApplyCurveFilter = False
        self._shouldRecolorRC = False
        self._shouldRecolorRGV = False
        self._shouldRecolorCMV = False
        self._shouldStrokeEdges = False
        self._shouldApplySharpenFilter = False
        self._shouldApplyFindEdgesFilter = False
        self._shouldApplyBlurFilter = False
        self._shouldApplyEmbossFilter = False
        self._shouldConvertBgr2Hsv = False

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
            # 顔を検出して・・・
            self._faceTracker.update(frame)

            # これが何をやっているのかサッパリわからん
            # faces = self._faceTracker.faces
            # rects.swapRects(frame, frame,
            #                 [face.faceRect for face in faces])

            # 検出した領域の周りに枠を描画する
            if self._shouldDrawDebugRects:
                self._faceTracker.drawDebugRects(frame)
            if self._shouldApplyCurveFilter:
                self._curveFilter.apply(frame, frame)
            if self._shouldRecolorRC:
                filters.recolorRC(frame, frame)
            if self._shouldRecolorRGV:
                filters.recolorRGV(frame, frame)
            if self._shouldRecolorCMV:
                filters.recolorCMV(frame, frame)
            if self._shouldStrokeEdges:
                filters.strokeEdges(frame, frame)
            if self._shouldApplySharpenFilter:
                self._sharpenFilter.apply(frame, frame)
            if self._shouldApplyFindEdgesFilter:
                self._findEdgesFilter.apply(frame, frame)
            if self._shouldApplyBlurFilter:
                self._blurFilter.apply(frame, frame)
            if self._shouldApplyEmbossFilter:
                self._embossFilter.apply(frame, frame)
            if self._shouldConvertBgr2Hsv:
                filters.convertBgr2Hsv(frame, frame)

            # フレームを解放する
            self._captureManager.exitFrame()
            # キーイベントがあれば実行する
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """
        キー入力を処理する
        スペース　：スクリーンショットを撮る
        タブ　　　：スクリーンキャストの録画を開始／終了する
        エスケープ：終了する
        :param keycode: int
        :return:
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
        elif keycode == ord('c'):
            self._shouldApplyCurveFilter = \
                not self._shouldApplyCurveFilter
        elif keycode == ord('d'):
            self._shouldDrawDebugRects = \
                not self._shouldDrawDebugRects
        elif keycode == ord('r'):
            self._shouldRecolorRC = \
                not self._shouldRecolorRC
        elif keycode == ord('v'):
            self._shouldRecolorRGV = \
                not self._shouldRecolorRGV
        elif keycode == ord('m'):
            self._shouldRecolorCMV = \
                not self._shouldRecolorCMV
        elif keycode == ord('s'):
            self._shouldStrokeEdges = \
                not self._shouldStrokeEdges
        elif keycode == ord('s'):
            self._shouldApplySharpenFilter = \
                not self._shouldApplySharpenFilter
        elif keycode == ord('f'):
            self._shouldApplyFindEdgesFilter = \
                not self._shouldApplyFindEdgesFilter
        elif keycode == ord('b'):
            self._shouldApplyBlurFilter = \
                not self._shouldApplyBlurFilter
        elif keycode == ord('e'):
            self._shouldApplyEmbossFilter = \
                not self._shouldApplyEmbossFilter
        elif keycode == ord('h'):
            self._shouldConvertBgr2Hsv = \
                not self._shouldConvertBgr2Hsv

if __name__=="__main__":
    Cameo().run()