# coding=utf-8
__author__ = 'weed'

import cv2
from datetime import datetime
import copy

import filters
from managers import WindowManager, CaptureManager
from trackers import FaceTracker

class Cameo(object):

    ADJUSTING = (
        HUE,
        HUE_RANGE,
        HOUGH_CIRCLE_RESOLUTION,
        HOUGH_CIRCLE_THRESHOLD,
        GAMMA
    ) = range(0, 5)

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
        self._shouldMaskByHue              = True
        self._hue                          = 140  # 緑
        self._hueRange                     = 60
        self._shouldEqualizeHist           = False
        self._shouldPaintBackgroundBlack   = False
        self._shouldProcessGaussianBlur    = True
        self._shouldProcessClosing         = True
        self._sThreshold                   = 5
        self._gamma                        = 100

        self._timeSelfTimerStarted         = None

        self._shouldDrawDebugRects         = False

        ### Ball Tracking ###
        self._shouldFindCircle             = False
        self._houghCircleDp                = 3
        self._houghCircleParam2            = 200
        self._centerPointOfCircle          = None
        self._passedPoints                 = []
        self._numberOfDisplayedPoints      = 50

        self._currentAdjusting             = self.HUE

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
                                  self._shouldProcessClosing, 1,
                                  self._sThreshold, self._gamma)

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
                                  self._shouldProcessGaussianBlur,
                                  True,  # Paint Background Black
                                  self._shouldProcessClosing)
                # グレースケール画像に変換する
                frameToFindCircle = cv2.cvtColor(frameToFindCircle, cv2.cv.CV_BGR2GRAY)
                # Hough変換で円を検出する
                height, width = frameToFindCircle.shape
                circles = cv2.HoughCircles(frameToFindCircle, cv2.cv.CV_HOUGH_GRADIENT, self._houghCircleDp,
                                           height / 4,  100, self._houghCircleParam2, 100, 1)
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

                # 円を描画する
                # もし円を見つけたら・・・
                if circles is not None:
                    x, y, r = circles[0][0]
                    self._centerPointOfCircle = (x,y)

                    # 円を描く
                    cv2.circle(frame, self._centerPointOfCircle, r, (0,255,0), 5)

                # 軌跡を描画する
                # 最初に円が見つかったときに初期化する
                if len(self._passedPoints) == 0 \
                        and self._centerPointOfCircle is not None:

                    # 最初の(x,y)でリストを埋める
                    for i in range(self._numberOfDisplayedPoints):
                        self._passedPoints.append(self._centerPointOfCircle)

                # 次の円を検出したら・・・
                elif self._centerPointOfCircle is not None:
                    # 通過点リストの最後に要素を追加する
                    self._passedPoints.append(self._centerPointOfCircle)
                    self._passedPoints.pop(0)  # 最初の要素は削除する

                # 次の円が見つかっても見つからなくても・・・
                if len(self._passedPoints) != 0:
                    for i in range(self._numberOfDisplayedPoints - 1):
                        # 軌跡を描画する
                        cv2.line(frame, self._passedPoints[i],
                                 self._passedPoints[i+1], (0,255,0), 5)

            # 情報を表示する
            def putText(text):
                cv2.putText(frame, text, (100,100),
                            cv2.FONT_HERSHEY_PLAIN, 2.0, (255,255,255), 3)

            adjusting = 'Adjusting '
            if   self._currentAdjusting == self.HUE:
                self.putText(adjusting + 'Hue')
            elif self._currentAdjusting == self.HUE_RANGE:
                self.putText(adjusting + 'Hue Range')
            elif self._currentAdjusting == self.HOUGH_CIRCLE_RESOLUTION:
                self.putText(adjusting + 'Hough Circle Resolution')
            elif self._currentAdjusting == self.HOUGH_CIRCLE_THRESHOLD:
                self.putText(adjusting + 'Hough Circle Threshold')
            elif self._currentAdjusting == self.GAMMA:
                self.putText(adjusting + 'Gamma')

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

        def _increaseParam(currentAdjusting, param, shouldIncrease):
            print id(param)
            pitch = paramDic[currentAdjusting]['pitch']
            if shouldIncrease:
                param += pitch
            else:
                param -= pitch
            print id(param)
            print paramDic[currentAdjusting]['name'] + ': ' + str(param)

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

        ### Adjustment
        elif keycode == ord('a'):
            self._isAdjustingHue = \
                not self._isAdjustingHue

        ### その他
        elif keycode == ord('d'):
            self._shouldDrawDebugRects = \
                not self._shouldDrawDebugRects
        elif keycode == ord('f'):
            self._shouldFindCircle = \
                not self._shouldFindCircle

        elif keycode == ord('p'):
             self._timeSelfTimerStarted = datetime.now()

        ### Adjustment
        elif keycode == 3:  # right arrow
            if not self._currentAdjusting == len(self.ADJUSTING) - 1:
                self._currentAdjusting += 1
        elif keycode == 2:  # left arrow
            if not self._currentAdjusting == 0:
                self._currentAdjusting -= 1
        elif keycode == 0 or keycode == 1:  # up / down arrow
            if self._currentAdjusting   == self.HUE:
                pitch = 10
                if keycode == 1:
                    pitch = - pitch
                self._hue               += pitch
                print 'hue: ' + str(self._hue)
            elif self._currentAdjusting == self.HUE_RANGE:
                pitch = 10
                if keycode == 1:
                    pitch = - pitch
                self._hueRange          += pitch
                print 'hueRange: ' + str(self._hueRange)
            elif self._currentAdjusting == self.HOUGH_CIRCLE_RESOLUTION:
                pitch = 1
                if keycode == 1:
                    pitch = - pitch
                self._houghCircleDp     += pitch
                print 'houghCircleDp: ' + str(self._houghCircleDp)
            elif self._currentAdjusting == self.HOUGH_CIRCLE_THRESHOLD:
                pitch = 50
                if keycode == 1:
                    pitch = - pitch
                self._houghCircleParam2 += pitch
                print 'houghCircleParam2: ' + str(self._houghCircleParam2)
            elif self._currentAdjusting == self.GAMMA:
                pitch = 10
                if keycode == 1:
                    pitch = - pitch
                self._gamma             += pitch
                print 'gamma: ' + str(self._gamma)
            else:
                raise ValueError('self._currentAdjusting')

        else:
            print keycode

if __name__ == "__main__":
    Cameo().run()