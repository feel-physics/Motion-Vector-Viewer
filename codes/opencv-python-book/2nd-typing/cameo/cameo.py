# coding=utf-8
__author__ = 'weed'

import cv2
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

        self._faceTracker = FaceTracker()
        self._shouldDrawDebugRects = True

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

            # TODO: フレームをフィルタ処理する（第3章）

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

if __name__=="__main__":
    Cameo().run()