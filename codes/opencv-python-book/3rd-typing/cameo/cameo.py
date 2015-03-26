# coding=utf-8
__author__ = 'weed'

import cv2
from datetime import datetime
import numpy
import math

import filters
from managers import WindowManager, CaptureManager

class Cameo(object):

    # TODO: 不要になったオプションは廃止する
    ADJUSTING_OPTIONS = (
        HUE_MIN,
        HUE_MAX,
        VALUE_MIN,
        VALUE_MAX,
        HOUGH_CIRCLE_RESOLUTION,
        HOUGH_CIRCLE_THRESHOLD,
        GAMMA,
        GAUSSIAN_BLUR_KERNEL_SIZE,
        SHOULD_TRACK_CIRCLE,
        SHOULD_PROCESS_GAUSSIAN_BLUR,
        SHOULD_PAINT_BACKGROUND_BLACK,
        SHOULD_PROCESS_CLOSING,
        CLOSING_ITERATIONS,
        SHOULD_DRAW_CIRCLE,
        SHOULD_DRAW_TRACKS,
        SHOULD_DRAW_VEROCITY_VECTOR,
        SHOWING_FRAME
    ) = range(0, 17)

    SHOWING_FRAME_OPTIONS = (
        ORIGINAL,
        MASKED_BY_HUE,  # TODO: 名前を変える
        WHAT_COMPUTER_SEE
    ) = range(0, 3)

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

        ### Filtering
        self._shouldMaskByHue              = False
        self._hueMin                       = 50  # 硬式テニスボール
        self._hueMax                       = 80
        self._valueMin                     = 80
        self._valueMax                     = 230
        self._shouldPaintBackgroundBlack   = False
        self._shouldProcessGaussianBlur    = True
        self._shouldProcessClosing         = True
        self._closingIterations            = 2
        self._sThreshold                   = 5
        self._gamma                        = 100
        self._gaussianBlurKernelSize       = 5
        self._shouldShowWhatComputerSee    = True

        self._timeSelfTimerStarted         = None

        ### Ball Tracking ###
        self._shouldTrackCircle            = False  # TODO: この変数は廃止する
        self._houghCircleDp                = 4
        self._houghCircleParam2            = 200
        self._centerPointOfCircle          = None
        self._passedPoints                 = []
        self._shouldDrawCircle             = False
        self._shouldDrawTracks             = False
        self._shouldDrawVerocityVector     = False
        self._lengthTimesOfVerocityVector  = 3

        self._currentAdjusting             = self.VALUE_MIN
        self._currentShowing               = self.MASKED_BY_HUE

    def _takeScreenShot(self):
        self._captureManager.writeImage(
            datetime.now().strftime('%y%m%d-%H%M%S')
            + '-screenshot.png')
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
            frameToDisplay = self._captureManager.frame
            frameToFindCircle = frameToDisplay.copy()  # 検出用のフレーム（ディープコピー）

            ### 画面表示

            def _processFrameToFindCircle(self, frame):
                """
                後で円を検出するために、検出用フレームに対して色相フィルタやぼかしなどの処理をする。
                SHOWING_WHAT_COMPUTER_SEEのときは、表示用フレームに対しても同じ処理をする。
                """
                filters.maskByHsv(frame, frame,
                                  self._hueMin, self._hueMax,
                                  self._valueMin, self._valueMax,
                                  True, self._gamma, self._sThreshold,
                                  self._shouldProcessGaussianBlur, self._gaussianBlurKernelSize,
                                  self._shouldProcessClosing, self._closingIterations)
                # グレースケール画像に変換する
                frameMasked_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return frameMasked_gray

            # TODO: 一つにまとめる
            if self._currentShowing == self.MASKED_BY_HUE:
                # def maskByHue(src, dst, hue, hueRange,
                #               shouldProcessGaussianBlur=False, shouldPaintBackgroundBlack=False,
                #               shouldProcessClosing=True, iterations=1):
                filters.maskByHsv(frameToDisplay, frameToDisplay,
                                  self._hueMin, self._hueMax,
                                  self._valueMin, self._valueMax,
                                  self._shouldPaintBackgroundBlack, self._gamma, self._sThreshold,
                                  self._shouldProcessGaussianBlur, self._gaussianBlurKernelSize,
                                  self._shouldProcessClosing, self._closingIterations)

            elif self._currentShowing == self.WHAT_COMPUTER_SEE:
                gray = _processFrameToFindCircle(self, frameToDisplay)
                cv2.merge((gray, gray, gray), frameToDisplay)

            # elif self._currentShowing == self.ORIGINAL:

            ### 検出・描画処理

            # 円を検出する
            if self._shouldTrackCircle:
                frameToFindCircle = _processFrameToFindCircle(self, frameToFindCircle)

                # Hough変換で円を検出する
                height, width = frameToFindCircle.shape
                circles = cv2.HoughCircles(frameToFindCircle, cv2.cv.CV_HOUGH_GRADIENT,
                                           self._houghCircleDp, height / 4,  100,
                                           self._houghCircleParam2, 100, 1)
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

                # もし円を見つけたら・・・
                if circles is not None:
                    # 中心座標と半径を取得して・・・
                    x, y, r = circles[0][0]
                    self._centerPointOfCircle = (x,y)

                    # 円を描く
                    if self._shouldDrawCircle:
                        cv2.circle(frameToDisplay, self._centerPointOfCircle, r, (0,255,0), 5)

                # 軌跡を描画する
                # 最初に円が見つかったときに初期化する
                # if len(self._passedPoints) == 0 \
                #         and self._centerPointOfCircle is not None:
                #
                #     # 最初の(x,y)でリストを埋める
                #     for i in range(self._numberOfDisplayedPoints):
                #         self._passedPoints.append(self._centerPointOfCircle)

                # 次の円を検出したら・・・
                if self._centerPointOfCircle is not None:
                    # 通過点リストの最後に要素を追加する
                    self._passedPoints.append(self._centerPointOfCircle)
                    # self._passedPoints.pop(0)  # 最初の要素は削除する

                # 次の円が見つかっても見つからなくても・・・
                if len(self._passedPoints) != 0:
                    numberOfPoints = len(self._passedPoints)
                    # 軌跡を描画する
                    if self._shouldDrawTracks:
                        if numberOfPoints > 1:
                            for i in range(numberOfPoints - 1):
                                cv2.line(frameToDisplay, self._passedPoints[i],
                                         self._passedPoints[i+1], (0,255,0), 5)
                    if self._shouldDrawVerocityVector:
                        if numberOfPoints > 2:
                            # 最後から1個前の点 pt0
                            pt0np = numpy.array(self._passedPoints[numberOfPoints - 2])
                            # 最後の点 pt1
                            pt1np = numpy.array(self._passedPoints[numberOfPoints - 1])
                            # 移動ベクトル Δpt = pt1 - pt0
                            dptnp = self._lengthTimesOfVerocityVector * (pt1np - pt0np)
                            areSamePoint_array = (dptnp == numpy.array([0,0]))
                            if not areSamePoint_array.all():
                                pt2np = pt1np + dptnp  # pt2 = pt1 + Δpt
                                pt1 = tuple(pt1np)
                                pt2 = tuple(pt2np)

                                def cvArrow(img, pt1, pt2, color, thickness=1, lineType=8, shift=0):
                                    cv2.line(img,pt1,pt2,color,thickness,lineType,shift)
                                    vx = pt2[0] - pt1[0]
                                    vy = pt2[1] - pt1[1]
                                    v  = math.sqrt(vx ** 2 + vy ** 2)
                                    ux = vx / v
                                    uy = vy / v
                                    # 矢印の幅の部分
                                    w = 5
                                    h = 10
                                    ptl = (int(pt2[0] - uy*w - ux*h), int(pt2[1] + ux*w - uy*h))
                                    ptr = (int(pt2[0] + uy*w - ux*h), int(pt2[1] - ux*w - uy*h))
                                    # 矢印の先端を描画する
                                    cv2.line(img,pt2,ptl,color,thickness,lineType,shift)
                                    cv2.line(img,pt2,ptr,color,thickness,lineType,shift)

                                cvArrow(frameToDisplay, pt1, pt2, (0,0,255), 5)

            ### 情報表示

            # 情報を表示する
            def _putText(text, lineNumber):
                cv2.putText(frameToDisplay, text, (100, 50 + 50 * lineNumber),
                            cv2.FONT_HERSHEY_PLAIN, 2.0, (255,255,255), 3)
            def _put(label, value):
                _putText(label, 1)
                if value is True:
                    value = 'True'
                elif value is False:
                    value = 'False'
                _putText(str(value), 2)

            if   self._currentAdjusting == self.HUE_MIN:
                _put('Hue Min'                      , self._hueMin)
            elif self._currentAdjusting == self.HUE_MAX:
                _put('Hue Max'                      , self._hueMax)
            elif self._currentAdjusting == self.VALUE_MIN:
                _put('Value Min'                    , self._valueMin)
            elif self._currentAdjusting == self.VALUE_MAX:
                _put('Value Max'                    , self._valueMax)
            elif self._currentAdjusting == self.HOUGH_CIRCLE_RESOLUTION:
                _put('Hough Circle Resolution'      , self._houghCircleDp)
            elif self._currentAdjusting == self.HOUGH_CIRCLE_THRESHOLD:
                _put('Hough Circle Threshold'       , self._houghCircleParam2)
            elif self._currentAdjusting == self.GAMMA:
                _put('Gamma'                        , self._gamma)
            elif self._currentAdjusting == self.GAUSSIAN_BLUR_KERNEL_SIZE:
                _put('Gaussian Blur Kernel Size'    , self._gaussianBlurKernelSize)
            elif self._currentAdjusting == self.SHOULD_TRACK_CIRCLE:
                _put('Should Track Circle'          , self._shouldTrackCircle)
            elif self._currentAdjusting == self.SHOULD_PROCESS_GAUSSIAN_BLUR:
                _put('Should Process Gaussian Blur' , self._shouldProcessGaussianBlur)
            elif self._currentAdjusting == self.SHOULD_PAINT_BACKGROUND_BLACK:
                _put('Should Paint Background Black', self._shouldPaintBackgroundBlack)
            elif self._currentAdjusting == self.SHOULD_PROCESS_CLOSING:
                _put('Should Process Closing'       , self._shouldProcessClosing)
            elif self._currentAdjusting == self.CLOSING_ITERATIONS:
                _put('Closing Iterations'           , self._closingIterations)
            elif self._currentAdjusting == self.SHOULD_DRAW_CIRCLE:
                _put('Should Draw Circle'           , self._shouldDrawCircle)
            elif self._currentAdjusting == self.SHOULD_DRAW_TRACKS:
                _put('Should Draw Tracks'           , self._shouldDrawTracks)
            elif self._currentAdjusting == self.SHOULD_DRAW_VEROCITY_VECTOR:
                _put('Should Draw Verocity Vector'  , self._shouldDrawVerocityVector)
            elif self._currentAdjusting == self.SHOWING_FRAME:
                if   self._currentShowing == self.ORIGINAL:
                    currentShowing = 'Original'
                elif self._currentShowing == self.MASKED_BY_HUE:
                    currentShowing = 'Masked By Hue'
                elif self._currentShowing == self.WHAT_COMPUTER_SEE:
                    currentShowing = 'What Computer See'
                else:
                    raise ValueError('self._currentShowing')

                _put('Showing Frame'                , currentShowing)
            else:
                raise ValueError('self._currentAdjusting')

            # フレームを解放する
            self._captureManager.exitFrame()
            # キーイベントがあれば実行する
            self._windowManager.processEvents()

            # セルフタイマー処理
            if self._timeSelfTimerStarted is not None:
                timeElapsed = datetime.now() - self._timeSelfTimerStarted
                # 3秒たったら・・・
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
                self._captureManager.startWritingVideo(
                    datetime.now().strftime('%y%m%d-%H%M%S')
                    + '-screencast.avi')
            # 書き出し中であれば・・・
            else:
                # ・・・書き出しを終える
                self._captureManager.stopWritingVideo()
        elif keycode == 27: # エスケープ
            self._windowManager.destroyWindow()

        ### Hue Filter ###
        elif keycode == ord('B'):
            self._hueMin = 200
            self._hueMax = 240
        elif keycode == ord('G'):
            self._hueMin = 80
            self._hueMax = 200
        elif keycode == ord('R'):
            self._hueMin = 0
            self._hueMax = 20
        elif keycode == ord('Y'):
            self._hueMin = 50
            self._hueMax = 80

        elif keycode == ord('p'):
             self._timeSelfTimerStarted = datetime.now()

        ### Adjustment
        elif keycode == 3:  # right arrow
            if not self._currentAdjusting == len(self.ADJUSTING_OPTIONS) - 1:
                self._currentAdjusting += 1
            else:
                self._currentAdjusting = 0
        elif keycode == 2:  # left arrow
            if not self._currentAdjusting == 0:
                self._currentAdjusting -= 1
            else:
                self._currentAdjusting = len(self.ADJUSTING_OPTIONS) - 1
        elif keycode == 0 or keycode == 1:  # up / down arrow
            if self._currentAdjusting   == self.HUE_MIN:
                pitch = 10 if keycode == 0 else -10
                self._hueMin            += pitch
            elif self._currentAdjusting == self.HUE_MAX:
                pitch = 10 if keycode == 0 else -10
                self._hueMax            += pitch
            elif self._currentAdjusting == self.VALUE_MIN:
                pitch = 10 if keycode == 0 else -10
                self._valueMin          += pitch
            elif self._currentAdjusting == self.VALUE_MAX:
                pitch = 10 if keycode == 0 else -10
                self._valueMax          += pitch
            elif self._currentAdjusting == self.HOUGH_CIRCLE_RESOLUTION:
                pitch = 1  if keycode == 0 else -1
                self._houghCircleDp     += pitch
            elif self._currentAdjusting == self.HOUGH_CIRCLE_THRESHOLD:
                pitch = 50 if keycode == 0 else -50
                self._houghCircleParam2 += pitch
            elif self._currentAdjusting == self.GAMMA:
                pitch = 10 if keycode == 0 else -10
                self._gamma             += pitch
            elif self._currentAdjusting == self.GAUSSIAN_BLUR_KERNEL_SIZE:
                pitch = 1  if keycode == 0 else -1
                self._gaussianBlurKernelSize += pitch
            elif self._currentAdjusting == self.SHOULD_TRACK_CIRCLE:
                self._passedPoints      =  []  # 軌跡を消去する
                self._shouldTrackCircle = \
                    not self._shouldTrackCircle
            elif self._currentAdjusting == self.SHOULD_PROCESS_GAUSSIAN_BLUR:
                self._shouldProcessGaussianBlur = \
                    not self._shouldProcessGaussianBlur
            elif self._currentAdjusting == self.SHOULD_PAINT_BACKGROUND_BLACK:
                if self._shouldPaintBackgroundBlack is False:
                    self._shouldMaskByHue = True
                    self._shouldPaintBackgroundBlack = True
                else:
                    self._shouldPaintBackgroundBlack = False
            elif self._currentAdjusting == self.SHOULD_PROCESS_CLOSING:
                self._shouldProcessClosing = \
                    not self._shouldProcessClosing
            elif self._currentAdjusting == self.CLOSING_ITERATIONS:
                pitch = 1  if keycode == 0 else -1
                self._closingIterations += pitch
            elif self._currentAdjusting == self.SHOULD_DRAW_CIRCLE:
                if self._shouldDrawCircle is False:
                    self._shouldTrackCircle = True
                    self._shouldDrawCircle = True
                else:
                    self._shouldDrawCircle = False
            elif self._currentAdjusting == self.SHOULD_DRAW_TRACKS:
                self._passedPoints = []  # 軌跡を消去する
                if self._shouldDrawTracks is False:
                    self._shouldTrackCircle = True
                    self._shouldDrawTracks = True
                else:
                    self._shouldDrawTracks = False
            elif self._currentAdjusting == self.SHOULD_DRAW_VEROCITY_VECTOR:
                self._passedPoints = []  # 軌跡を消去する
                if self._shouldDrawVerocityVector is False:
                    self._shouldTrackCircle = True
                    self._shouldDrawVerocityVector = True
                else:
                    self._shouldDrawVerocityVector = False
            elif self._currentAdjusting == self.SHOWING_FRAME:
                if   keycode == 0:  # up arrow
                    if not self._currentShowing == len(self.SHOWING_FRAME_OPTIONS) - 1:
                        self._currentShowing += 1
                    else:
                        self._currentShowing = 0
                elif keycode == 1:  # down arrow
                    if not self._currentShowing == 0:
                        self._currentShowing -= 1
                    else:
                        self._currentShowing = len(self.SHOWING_FRAME_OPTIONS) - 1

            else:
                raise ValueError('self._currentAdjusting')

        else:
            print keycode

if __name__ == "__main__":
    Cameo().run()