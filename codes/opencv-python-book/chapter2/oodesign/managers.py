# coding=utf-8
__author__ = 'weed'

import cv2
import numpy
import time

class CaptureManager(object):

    def __init__(self, capture, previewWindowManager = None,
                 shouldMirrorPreview = False):

        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview  = shouldMirrorPreview

        self._capture       = capture
        self._channel       = 0
        self._enteredFrame  = False
        self._frame         = None
        self._imageFileName = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter   = None

        self._startTime     = None
        self._framesElapsed = long(0)
        self._fpsEstimate   = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame   = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve(
                channel = self.channel
            )
        return self._frame

    @property
    def isWritingImage(self):

        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        return self._videoFilename is not None

    def enterFrame(self):
        """とにかく次のフレームをキャプチャーします"""

        # まず直前のフレームがあるかを確認します
        assert not self._enteredFrame, \
            '直前の enterFrame() が対応する exitFrame() を持っていません'

        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def exitFrame(self):
        """ウィンドウに描きます。ファイルに書きます。そのフレームは解放します。"""

        # つかんだフレームが取ってこれるか確認します
        # getterがそのフレームを取ってきてキャッシュするかもしれません
        if self.frame is None:
            self._enteredFrame = False
            return

        # FPSの予測値と関係する変数を更新します
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            self._fpsEstimate = self._framesElapsed / timeElapsed
        self._framesElapsed += 1

        # とにかくウィンドウに描きます
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = numpy.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self._frame)

        # とにかく画像ファイルに書き出します
        if self.isWritingImage:
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None

        # とにかく動画ファイルに書き出します
        self._writeVideoFrame()

        # そのフレームを解放します
        self._frame = None
        self._enteredFrame = False

    def writeImage(self, filename):
        """次のフレームを画像ファイルの中に書き出す"""
        self._imageFilename = filename

    def startWritingVideo(
            self, filename,
            encoding = cv2.cv.CV_FOURCC('I','4','2','0')
    ):
        """描画したフレームを動画ファイルに書き始める"""
        self._videoFilename = filename
        self._videoEncoding = encoding

    def stopWritingVideo(self):
        """フレームを動画ファイルに書き出すのを止める"""
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter   = None

    def _writeVideoFrame(self):

        if not self.isWritingVideo:
            return

        if self._videoWriter is None:
            fps = self._capture.get(cv2.cv.CV_CAP_PROP_FPS)
            if fps == 0.0:
                # キャプチャーのFPSがわからないため、予測したものを使う
                if self._framesElapsed < 20:
                    # 予測が安定するまでフレームを待つ
                    return
                else:
                    fps = self._fpsEstimate
            size = (
                int(self._capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH )),
                int(self._capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
            )
            self._videoWriter = cv2.VideoWriter(
                self._videoFilename, self._videoEncoding, fps, size
            )

        self._videoWriter.write(self._frame)

