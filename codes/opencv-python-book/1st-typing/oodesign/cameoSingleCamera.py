# coding=utf-8
__author__ = 'weed'

import cv2
from managers import WindowManager, CaptureManager
import rects
from trackers import FaceTracker

class Cameo(object):

    def __init__(self):
        self._windowManager  = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, True
        )
        self._faceTracker = FaceTracker()
        self._shouldDrawDebugRects = False

    def run(self):
        """メインループを走らせる"""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            # TODO: フレームをフィルタにかける

            self._faceTracker.update(frame)
            faces = self._faceTracker.faces
            rects.swapRects(frame, frame,
                            [face.faceRect for face in faces])

            if self._shouldDrawDebugRects:
                self._faceTracker.drawDebugRects(frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """
        キー入力を処理する

        スペース　：スクリーンショットを撮る
        タブ　　　：スクリーンキャストを録画開始／終了する
        ｘ　　　　：顔の周りのデバッグ用矩形を描画開始／終了する
        エスケープ：終了する
        """
        if keycode == 32: # スペース
            self._captureManager.writeImage('screen-shot.png')
        elif keycode == 9: # タブ
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screen-cast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 120: # x
            self._shouldDrawDebugRects = \
                not self._shouldDrawDebugRects
        elif keycode == 27: # エスケープ
            self._windowManager.destroyWindow()

if __name__=="__main__":
    Cameo().run()