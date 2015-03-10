# coding=utf-8
__author__ = 'weed'

import cv2
import filters
from managers import WindowManager, CaptureManager
from trackers import FaceTracker
from datetime import datetime


class Cameo(object):

    def __init__(self):

        self._windowManager = WindowManager('Cameo',
                                            self.onKeypress)
        """:type : managers.WindowManager"""
        # def __init__(self, windowName, keypressCallback = None):

        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, False)
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
        self._shouldRecolorRC              = False
        self._shouldRecolorRGV             = False
        self._shouldRecolorCMV             = False
        self._shouldMaskByHue              = False
        self._hue                          = 140  # 緑
        self._hueRange                     = 60
        self._shouldEqualizeHist           = False
        self._shouldPaintBackgroundBlack   = False
        self._shouldProcessGaussianBlur    = False
        self._shouldProcessClosing         = False

        self._timeSelfTimerStarted         = None

        self._shouldDrawDebugRects         = False

        self._shouldFindCircle             = False

    def _takeScreenShot(self):
        now = datetime.now()
        timestamp = now.strftime('%y%m%d-%H%M%S')
        self._captureManager.writeImage(timestamp + '-screen-shot.png')
        print 'captured'

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
            if self._shouldRecolorRC:
                filters.recolorRC(frame, frame)
            if self._shouldRecolorRGV:
                filters.recolorRGV(frame, frame)
            if self._shouldRecolorCMV:
                filters.recolorCMV(frame, frame)
            if self._shouldEqualizeHist:
                filters.equaliseHist(frame, frame)

            # def maskByHue(src, dst, hue, hueRange,
            #               shouldProcessGaussianBlur=False, shouldPaintBackgroundBlack=False,
            #               shouldProcessClosing=True, iterations=1):
            if self._shouldMaskByHue:
                filters.maskByHue(frame, frame, self._hue, self._hueRange,
                                  self._shouldProcessGaussianBlur,
                                  self._shouldPaintBackgroundBlack,
                                  self._shouldProcessClosing)

            # 検出した領域の周りに枠を描画する
            if self._shouldDrawDebugRects:
                # 顔を検出して・・・
                self._faceTracker.update(frame)

                # これが何をやっているのかサッパリわからん
                # faces = self._faceTracker.faces
                # rects.swapRects(frame, frame,
                #                 [face.faceRect for face in faces])

                self._faceTracker.drawDebugRects(frame)

            # 円を検出する
            if self._shouldFindCircle:
                frameToFindCircle = frame.copy()
                filters.maskByHue(frame, frameToFindCircle, self._hue, self._hueRange,
                                  self._shouldProcessGaussianBlur, True,
                                  self._shouldProcessClosing)
                # グレースケール画像に変換する
                frameToFindCircle = cv2.cvtColor(frameToFindCircle, cv2.cv.CV_BGR2GRAY)
                # Hough変換で円を検出する
                height, width = frameToFindCircle.shape
                circles = cv2.HoughCircles(frameToFindCircle, cv2.cv.CV_HOUGH_GRADIENT, 2,
                                           height / 4)
                # cv2.HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[,
                #                  minRadius[, maxRadius]]]]]) → circles
                # ハフ変換を用いて，グレースケール画像から円を検出します．
                # パラメタ:
                # image – 8ビット，シングルチャンネル，グレースケールの入力画像．
                # circles – 検出された円を出力するベクトル．
                #   各ベクトルは，3要素の浮動小数点型ベクトル  (x, y, radius) としてエンコードされます．
                # method – 現在のところ， CV_HOUGH_GRADIENT メソッドのみが実装されています．
                #   基本的には 2段階ハフ変換 で，これについては Yuen90 で述べられています．
                # dp – 画像分解能に対する投票分解能の比率の逆数．
                #   例えば， dp=1 の場合は，投票空間は入力画像と同じ分解能をもちます．
                #   また dp=2 の場合は，投票空間の幅と高さは半分になります．
                # minDist – 検出される円の中心同士の最小距離．
                #   このパラメータが小さすぎると，正しい円の周辺に別の円が複数誤って検出されることになります．
                #   逆に大きすぎると，検出できない円がでてくる可能性があります．
                # param1 – 手法依存の 1 番目のパラメータ．
                #   CV_HOUGH_GRADIENT の場合は，
                #   Canny() エッジ検出器に渡される2つの閾値の内，大きい方の閾値を表します
                #   （小さい閾値は，この値の半分になります）．
                # param2 – 手法依存の 2 番目のパラメータ．
                #   CV_HOUGH_GRADIENT の場合は，円の中心を検出する際の投票数の閾値を表します．
                #   これが小さくなるほど，より多くの誤検出が起こる可能性があります．
                #   より多くの投票を獲得した円が，最初に出力されます．
                # minRadius – 円の半径の最小値．
                # maxRadius – 円の半径の最大値．

                # 円を描く
                if circles is not None:
                    x, y, r = circles[0][0]
                    cv2.circle(frame, (x,y), r, (0,255,0), 5)

            # フレームを解放する
            self._captureManager.exitFrame()
            # キーイベントがあれば実行する
            self._windowManager.processEvents()

            if self._timeSelfTimerStarted is not None:
                timeElapsed = datetime.now() - self._timeSelfTimerStarted
                if timeElapsed.seconds > 3:
                    self._takeScreenShot()
                    # タイマーをリセットする
                    self._timeSelfTimerStarted = None

    def onKeypress(self, keycode):
        """
        キー入力を処理するe
        スペース　：スクリーンショットを撮る
        タブ　　　：スクリーンキャストの録画を開始／終了する
        エスケープ：終了する
        :param keycode: int
        :return: None
        """

        ### 基本操作
        if keycode == 32:  # スペース
            self._captureManager.paused = \
                not self._captureManager.paused
        elif keycode == 13:  # リターン
            self._takeScreenShot()

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

        ### Hue Filter ###
        elif keycode == ord('h'):
            self._shouldMaskByHue = \
                not self._shouldMaskByHue
        elif keycode == 0:  # up arrow
            self._hue += 10
            print 'hue     : ' + str(self._hue)
        elif keycode == 1:  # down arrow
            self._hue -= 10
            print 'hue     : ' + str(self._hue)
        elif keycode == 2:  # left arrow
            self._hueRange -= 10
            print 'hueRange: ' + str(self._hueRange)
        elif keycode == 3:  # right arrow
            self._hueRange += 10
            print 'hueRange: ' + str(self._hueRange)
        elif keycode == ord('B'):
            self._hue      = 220
            self._hueRange = 20
            self._shouldMaskByHue = \
                not self._shouldMaskByHue
        elif keycode == ord('G'):
            self._hue      = 140
            self._hueRange = 60
            self._shouldMaskByHue = \
                not self._shouldMaskByHue
        elif keycode == ord('R'):
            self._hue      = 10
            self._hueRange = 10
            self._shouldMaskByHue = \
                not self._shouldMaskByHue
        elif keycode == ord('Y'):
            self._hue      = 60
            self._hueRange = 30
            self._shouldMaskByHue = \
                not self._shouldMaskByHue
        elif keycode == ord('g'):
            self._shouldMaskByHue = True
            self._shouldProcessGaussianBlur = \
                not self._shouldProcessGaussianBlur
        elif keycode == ord('k'):
            self._shouldMaskByHue = True
            self._shouldPaintBackgroundBlack = \
                not self._shouldPaintBackgroundBlack
        elif keycode == ord('c'):
            self._shouldMaskByHue = True
            self._shouldProcessClosing = \
                not self._shouldProcessClosing

        ### その他
        elif keycode == ord('d'):
            self._shouldDrawDebugRects = \
                not self._shouldDrawDebugRects
        elif keycode == ord('f'):
            self._shouldFindCircle = \
                not self._shouldFindCircle

        elif keycode == ord('p'):
             self._timeSelfTimerStarted = datetime.now()

        else:
            print keycode

if __name__ == "__main__":
    Cameo().run()