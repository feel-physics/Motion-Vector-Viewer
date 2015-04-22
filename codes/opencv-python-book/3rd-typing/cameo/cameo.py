# coding=utf-8
__author__ = 'weed'

import cv2
from datetime import datetime
import numpy

import filters
from managers import WindowManager, CaptureManager
import utils

class Cameo(object):

    ##### TODO: 不要になったオプションは廃止する
    ADJUSTING_OPTIONS = (
        HUE_MIN,
        HUE_MAX,
        VALUE_MIN,
        VALUE_MAX,
        SHOULD_PROCESS_GAUSSIAN_BLUR,
        GAUSSIAN_BLUR_KERNEL_SIZE,
        SHOULD_PROCESS_CLOSING,
        CLOSING_ITERATIONS,
        HOUGH_CIRCLE_RESOLUTION,
        HOUGH_CIRCLE_CANNY_THRESHOLD,
        HOUGH_CIRCLE_ACCUMULATOR_THRESHOLD,
        SHOULD_DRAW_CANNY_EDGE,
        SHOULD_DRAW_CIRCLE,
        SHOULD_DRAW_TRACKS,
        SHOULD_DRAW_DISPLACEMENT_VECTOR,
        SHOULD_DRAW_VEROCITY_VECTOR,
        SHOULD_DRAW_ACCELERATION_VECTOR,
        GRAVITY_STRENGTH,
        SHOULD_DRAW_FORCE_VECTOR,
        SHOULD_DRAW_SYNTHESIZED_VECTOR,
        SHOULD_TRACK_CIRCLE,
        SHOWING_FRAME
    ) = range(0, 22)

    SHOWING_FRAME_OPTIONS = (
        ORIGINAL,
        GRAY_SCALE,
        WHAT_COMPUTER_SEE
    ) = range(0, 3)

    def __init__(self):

        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, False)

        ### Filtering
        self._hueMin                       = 50  # 硬式テニスボール
        self._hueMax                       = 80
        self._sThreshold                   = 5
        self._valueMin                     = 60
        self._valueMax                     = 260
        self._gamma                        = 100
        self._shouldProcessGaussianBlur    = True
        self._gaussianBlurKernelSize       = 20
        self._shouldProcessClosing         = True
        self._closingIterations            = 2

        ### Ball Tracking ###
        self._houghCircleDp                = 4
        self._houghCircleParam1            = 100
        self._houghCircleParam2            = 150
        self._shouldDrawCannyEdge          = False

        self._centerPointOfCircle          = None
        self._passedPoints                 = []

        self._shouldTrackCircle            = True
        self._isTracking                   = False
        self._track_window                 = None
        self._roi_hist                     = None

        self._shouldDrawCircle             = False
        self._shouldDrawTracks             = False
        self._shouldDrawDisplacementVector = False
        self._shouldDrawVerocityVector     = False
        self._shouldDrawAccelerationVector = True
        self._shouldDrawForceVector        = False
        self._gravityStrength              = 100
        self._shouldDrawSynthesizedVector  = False

        self._currentAdjusting             = self.GRAVITY_STRENGTH
        self._currentShowing               = self.ORIGINAL

        self._numFramesDelay               = 6
        self._enteredFrames                = []

        self._timeSelfTimerStarted         = None
        self._timeArrayToCalcFps           = []

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
        # FPS計算用の時刻
        self._timeArrayToCalcFps.append(datetime.now())
        # ウィンドウが存在する限り・・・
        while self._windowManager.isWindowCreated:
            # フレームを取得し・・・
            self._captureManager.enterFrame()
            frameToDisplay = self._captureManager.frame

            # 数フレームを配列にためて、
            # 新しいフレームを末尾に追加し、
            # 最初のフレームを取り出して表示する。
            frameNow = frameToDisplay.copy()  # 本当の現在のフレーム
            self._enteredFrames.append(frameToDisplay.copy())  # ディープコピーしないと参照を持って行かれる
            frameToDisplay[:] = self._enteredFrames[0]  # ためたフレームの最初のものを表示する
            if len(self._enteredFrames) <= self._numFramesDelay:  # 最初はためる
                pass
            else:
                self._enteredFrames.pop(0)  # たまったら最初のものは削除していく

            # frameToFindCircle = frameToDisplay.copy()  # 検出用のフレーム（ディープコピー）


            ### 画面表示 ###


            def getMaskToFindCircle(self, frame):
                """
                後で円を検出するために、検出用フレームに対して色相フィルタやぼかしなどの処理をする。
                SHOWING_WHAT_COMPUTER_SEEのときは、表示用フレームに対しても同じ処理をする。
                """
                mask = filters.getMaskByHsv(frame, self._hueMin, self._hueMax, self._valueMin, self._valueMax,
                                            self._gamma, self._sThreshold, self._shouldProcessGaussianBlur,
                                            self._gaussianBlurKernelSize, self._shouldProcessClosing,
                                            self._closingIterations)
                return mask

            def getCircles(self, frame):
                """
                Hough変換で円を検出する
                :return: 検出した円のx,y,r
                """
                height, width = frame.shape
                circles = cv2.HoughCircles(
                    frame,        # 画像
                    cv2.cv.CV_HOUGH_GRADIENT, # アルゴリズムの指定
                    self._houghCircleDp,      # 内部でアキュムレーションに使う画像の分解能(入力画像の解像度に対する逆比)
                    width / 10,               # 円同士の間の最小距離
                    self._houghCircleParam1,  # 内部のエッジ検出(Canny)で使う閾値
                    self._houghCircleParam2,  # 内部のアキュムレーション処理で使う閾値
                    100,                      # 円の最小半径
                    1)                        # 円の最大半径
                return circles

            if self._currentShowing == self.GRAY_SCALE:
                mask = getMaskToFindCircle(self, frameToDisplay)

                # カメラ画像をHSVチャンネルに分離し・・・
                frame = cv2.cvtColor(frameToDisplay, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(frame)

                # マスク部分の明度をガンマ補正し・・・
                v = filters.letMaskMoreBright(v, mask, self._gamma)

                # マスク部分以外は・・・

                # mask（1チャンネル画像）の該当ピクセルが0のとき、
                # notMask（1チャンネル画像）の該当ピクセルを255にセットする。
                # さもなくば、0にセットする。
                # 要するにnotMaskはmaskを反転させたもの。
                notMask = cv2.compare(mask, 0, cv2.CMP_EQ)

                # 彩度を0にする
                cv2.bitwise_and(s, 0, s, notMask) # 論理積

                frame = cv2.merge((h, s, v))
                cv2.cvtColor(frame, cv2.COLOR_HSV2BGR, frameToDisplay)

            elif self._currentShowing == self.WHAT_COMPUTER_SEE:
                pass
                # gray = getMaskToFindCircle(self, frameToDisplay)
                # cv2.merge((gray, gray, gray), frameToDisplay)

            elif self._currentShowing == self.ORIGINAL:
                pass


            ### 物体追跡・描画処理 ###


            if self._shouldTrackCircle and not self._isTracking:
                # 検出用フレームをつくる
                frameToFindCircle = getMaskToFindCircle(self, frameNow)
                circles = getCircles(self, frameToFindCircle)  # 円を検出する

                # TODO:ここが動かない
                # if self._currentShowing == self.WHAT_COMPUTER_SEE:
                #     gray = getMaskToFindCircle(self, frameToDisplay)
                #     cv2.merge((gray, gray, gray), frameToDisplay)

                if circles is not None:  # もし円を見つけたら・・・
                    x, y, r = circles[0][0]  # 中心座標と半径を取得して・・・
                    x, y ,r = int(x), int(y), int(r)  # 整数にする
                    # 画面外に円がはみ出す場合は・・・
                    height, width = frameToFindCircle.shape
                    m = 10  # マージン
                    if x < r+m or width < x+r+m or y < r+m or height < y+r+m:
                        pass
                    # 画面の中に円が収まる場合は・・・
                    else:
                        # 追跡したい領域の初期設定
                        self._track_window = (x-r, y-r, 2*r, 2*r)
                        # 追跡のためのROI関心領域（Region of Interest)を設定
                        roi = frameNow[y-r:y+r, x-r:x+r]
                        # HSV色空間に変換
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        # マスク画像の作成
                        mask = cv2.inRange(hsv_roi, numpy.array((
                            self._hueMin / 2,           # H最小値
                            2 ** self._sThreshold - 1,  # S最小値
                            self._valueMin              # V最小値
                        )), numpy.array((
                            self._hueMax / 2,           # H最大値
                            255,                        # S最大値
                            self._valueMax)))           # V最大値
                        # ヒストグラムの計算
                        self._roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
                        # ヒストグラムの正規化
                        cv2.normalize(self._roi_hist,self._roi_hist,0,255,cv2.NORM_MINMAX)

                        # # maskを描画するコード（デバッグ用）
                        # mask3Channel = cv2.merge((mask, mask, mask))
                        # if mask is not None and self._track_window is not None:
                        #     utils.pasteRect(frameToDisplay, frameToDisplay, mask3Channel, self._track_window)

                        self._isTracking = True

            # if self._shouldTrackCircle and not self._isTracking:
            elif self._shouldTrackCircle and self._isTracking:
                # HSV色空間に変換
                hsv = cv2.cvtColor(frameNow, cv2.COLOR_BGR2HSV)
                # バックプロジェクションの計算
                dst = cv2.calcBackProject([hsv],[0],self._roi_hist,[0,180],1)

                # バックプロジェクションを描画するコード（デバッグ用）
                if self._currentShowing == self.WHAT_COMPUTER_SEE:
                    cv2.merge((dst, dst, dst), frameToDisplay)

                # 新しい場所を取得するためにmeanshiftを適用
                ret, self._track_window = cv2.meanShift(dst, self._track_window,
                                                        ( cv2.TERM_CRITERIA_EPS |
                                                          cv2.TERM_CRITERIA_COUNT, 10, 1 ))
                x,y,w,h = self._track_window

                # 追跡している領域を描く
                if self._currentShowing == self.WHAT_COMPUTER_SEE:
                    cv2.rectangle(frameToDisplay, (x,y), (x+w,y+h),(0,0,200),5)

                # 次の円を検出したら・・・
                if self._track_window is not None:
                    # 通過点リストの最後に要素を追加する
                    self._passedPoints.append((x+w/2, y+h/2))
                    # self._passedPoints.pop(0)  # 最初の要素は削除する

                # 次の円が見つかっても見つからなくても・・・
                if len(self._passedPoints) - self._numFramesDelay > 0:
                    numPointsVisible = len(self._passedPoints) - self._numFramesDelay

                    # if numPointsVisible >= 2:
                    #     print self._passedPoints[numPointsVisible-1][0] \
                    #         - self._passedPoints[numPointsVisible-2][0], \
                    #         + self._passedPoints[numPointsVisible-1][1] \
                    #         - self._passedPoints[numPointsVisible-2][1]


                    # 軌跡を描画する
                    if self._shouldDrawTracks:
                        for i in range(numPointsVisible - 1):
                            cv2.line(frameToDisplay, self._passedPoints[i],
                                     self._passedPoints[i+1], (255,255,255), 5)
                            # # 軌跡ではなく打点する（デバッグ用）
                            # cv2.circle(frameToDisplay, self._passedPoints[i], 1, (255,255,255), 5)

                    pt = self._passedPoints[numPointsVisible]

                    # 変位ベクトルを描画する
                    if self._shouldDrawDisplacementVector:
                        vector = (pt[0] - self._passedPoints[0][0],
                                  pt[1] - self._passedPoints[0][1])
                        if vector is not None:
                            utils.cvArrow(frameToDisplay, self._passedPoints[0],
                                          vector, 1, (255,255,255), 5)

                    # 速度ベクトルを描画する
                    if self._shouldDrawVerocityVector:
                        vector = utils.getVelocityVector(self._passedPoints, self._numFramesDelay,
                                                         int(self._numFramesDelay/2))
                        if vector is not None:
                            utils.cvArrow(frameToDisplay, pt, vector, 4, (255,0,0), 5)

                    # 加速度ベクトルを描画する
                    if self._shouldDrawAccelerationVector:
                        # vector = utils.getAccelerationVector(self._passedPoints, self._numFramesDelay*2)
                        # vector = utils.getAccelerationVectorVelocitySensitive(self._passedPoints)
                        vector = utils.getAccelerationVectorFirFilter(self._passedPoints, 6, 6)
                        if vector is not None:
                            utils.cvArrow(frameToDisplay, pt, vector, 1, (0,255,0), 5)

                    # 力ベクトルを描画する
                    yPtAcl = self._passedPoints[numPointsVisible][1] + h/2
                    ptAcl = (self._passedPoints[numPointsVisible][0], yPtAcl)
                    if self._shouldDrawForceVector:
                        aclVector = utils.getAccelerationVector(self._passedPoints, self._numFramesDelay*2)
                        if aclVector is None:
                            aclVector = (0,0)
                        vector = (aclVector[0], aclVector[1] - self._gravityStrength)

                        if vector is not None:
                            utils.cvArrow(frameToDisplay, ptAcl, vector, 1, (0,0,255), 5)

                    # 力ベクトルの合成を描画する
                    if self._shouldDrawSynthesizedVector:
                        # 手による接触力
                        aclVector = utils.getAccelerationVector(self._passedPoints, self._numFramesDelay*2)
                        if aclVector is None:
                            aclVector = (0,0)
                        contactForceVector = (aclVector[0], aclVector[1] - self._gravityStrength)
                        if contactForceVector is not None:
                            utils.cvArrow(frameToDisplay, pt, contactForceVector, 1, (0,0,255), 2)
                        # 重力
                        gravityForceVector = (0, self._gravityStrength)
                        utils.cvArrow(frameToDisplay, pt, gravityForceVector, 1, (0,0,255), 2)
                        # 合力
                        synthesizedVector = utils.getAccelerationVector(self._passedPoints,
                                                                        self._numFramesDelay*2)
                        if synthesizedVector is not None:
                            utils.cvArrow(frameToDisplay, pt, synthesizedVector, 1, (0,0,255), 5)
                            # 接触力ベクトルと加速度ベクトルのあいだに線を引く
                            ptSV = (pt[0]+synthesizedVector[0], pt[1]+synthesizedVector[1])
                            if contactForceVector is not None:
                                ptCF = (pt[0]+contactForceVector[0], pt[1]+contactForceVector[1])
                                utils.cvLine(frameToDisplay, ptSV, ptCF, (0,0,255), 1)
                            # 重力ベクトルと加速度ベクトルのあいだに線を引く
                            ptGF = (pt[0], pt[1]+self._gravityStrength)
                            utils.cvLine(frameToDisplay, ptSV, ptGF, (0,0,255), 1)



            # Cannyエッジ検出
            if self._shouldDrawCannyEdge:
                gray = cv2.cvtColor(frameToDisplay, cv2.COLOR_BGR2GRAY)
                edge = cv2.Canny(gray, self._houghCircleParam1/2, self._houghCircleParam1)
                cv2.merge((edge, edge, edge), frameToDisplay)


            ### 情報表示 ###


            # FPSを計算する
            if len(self._timeArrayToCalcFps) < 10:
                self._timeArrayToCalcFps.append(datetime.now())
                fps = -1
            else:
                self._timeArrayToCalcFps.append(datetime.now())
                self._timeArrayToCalcFps.pop(0)
                timeElapsed = self._timeArrayToCalcFps[9] - self._timeArrayToCalcFps[0]
                fps = 10 / (timeElapsed.seconds + timeElapsed.microseconds / 1000000.0)

            # 情報を表示する
            def putText(text, lineNumber):
                cv2.putText(frameToDisplay, text, (100, 50 + 50 * lineNumber),
                            cv2.FONT_HERSHEY_PLAIN, 2.0, (255,255,255), 3)
            def put(label, value):
                putText('FPS ' + "{0:.1f}".format(fps), 1)
                putText(label, 2)
                if value is True:
                    value = 'True'
                elif value is False:
                    value = 'False'
                putText(str(value), 3)

            cur = self._currentAdjusting

            if   cur == self.HUE_MIN:
                put('Hue Min'                            , self._hueMin)
            elif cur == self.HUE_MAX:
                put('Hue Max'                            , self._hueMax)
            elif cur == self.VALUE_MIN:
                put('Value Min'                          , self._valueMin)
            elif cur == self.VALUE_MAX:
                put('Value Max'                          , self._valueMax)
            elif cur == self.HOUGH_CIRCLE_RESOLUTION:
                put('Hough Circle Resolution'            , self._houghCircleDp)
            elif cur == self.HOUGH_CIRCLE_CANNY_THRESHOLD:
                put('Hough Circle Canny Threshold'       , self._houghCircleParam1)
            elif cur == self.HOUGH_CIRCLE_ACCUMULATOR_THRESHOLD:
                put('Hough Circle Accumulator Threshold' , self._houghCircleParam2)
            elif cur == self.GAUSSIAN_BLUR_KERNEL_SIZE:
                put('Gaussian Blur Kernel Size'          , self._gaussianBlurKernelSize)
            elif cur == self.SHOULD_PROCESS_GAUSSIAN_BLUR:
                put('Process Gaussian Blur'              , self._shouldProcessGaussianBlur)
            elif cur == self.SHOULD_PROCESS_CLOSING:
                put('Process Closing'                    , self._shouldProcessClosing)
            elif cur == self.CLOSING_ITERATIONS:
                put('Closing Iterations'                 , self._closingIterations)
            elif cur == self.SHOULD_DRAW_CIRCLE:
                put('Should Draw Circle'                 , self._shouldDrawCircle)
            elif cur == self.SHOULD_DRAW_TRACKS:
                put('Should Draw Tracks'                 , self._shouldDrawTracks)
            elif cur == self.SHOULD_DRAW_DISPLACEMENT_VECTOR:
                put('Should Draw Displacement Vector'    , self._shouldDrawDisplacementVector)
            elif cur == self.SHOULD_DRAW_VEROCITY_VECTOR:
                put('Should Draw Verocity Vector'        , self._shouldDrawVerocityVector)
            elif cur == self.SHOULD_DRAW_ACCELERATION_VECTOR:
                put('Should Draw Acceleration Vector'    , self._shouldDrawAccelerationVector)
            elif cur == self.GRAVITY_STRENGTH:
                put('Gravity Strength'                   , self._gravityStrength)
            elif cur == self.SHOULD_DRAW_FORCE_VECTOR:
                put('Should Draw Force Vector'           , self._shouldDrawForceVector)
            elif cur == self.SHOULD_DRAW_SYNTHESIZED_VECTOR:
                put('Should Draw Synthesized Vector'     , self._shouldDrawSynthesizedVector)
            elif cur == self.SHOULD_TRACK_CIRCLE:
                put('Should Track Circle'                , self._shouldTrackCircle)
            elif cur == self.SHOULD_DRAW_CANNY_EDGE:
                put('Should Draw Canny Edge'             , self._shouldDrawCannyEdge)
            elif cur == self.SHOWING_FRAME:
                if   self._currentShowing == self.ORIGINAL:
                    currentShowing = 'Original'
                elif self._currentShowing == self.GRAY_SCALE:
                    currentShowing = 'Gray Scale'
                elif self._currentShowing == self.WHAT_COMPUTER_SEE:
                    currentShowing = 'What Computer See'
                else:
                    raise ValueError('self._currentShowing')

                put('Showing Frame'                , currentShowing)
            else:
                raise ValueError('self._currentAdjusting')


            ### 1フレーム終了 ###


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


        ### 基本操作 ###


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


        ### Adjustment ###


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
            elif self._currentAdjusting == self.HOUGH_CIRCLE_CANNY_THRESHOLD:
                pitch = 20 if keycode == 0 else -20
                self._houghCircleParam1 += pitch
            elif self._currentAdjusting == self.HOUGH_CIRCLE_ACCUMULATOR_THRESHOLD:
                pitch = 50 if keycode == 0 else -50
                self._houghCircleParam2 += pitch
            elif self._currentAdjusting == self.GAUSSIAN_BLUR_KERNEL_SIZE:
                pitch = 1  if keycode == 0 else -1
                self._gaussianBlurKernelSize += pitch
            elif self._currentAdjusting == self.SHOULD_PROCESS_GAUSSIAN_BLUR:
                self._shouldProcessGaussianBlur = \
                    not self._shouldProcessGaussianBlur
            elif self._currentAdjusting == self.SHOULD_PROCESS_CLOSING:
                self._shouldProcessClosing = \
                    not self._shouldProcessClosing
            elif self._currentAdjusting == self.CLOSING_ITERATIONS:
                pitch = 1  if keycode == 0 else -1
                self._closingIterations += pitch
            elif self._currentAdjusting == self.SHOULD_DRAW_CIRCLE:
                if  self._shouldDrawCircle:
                    self._shouldDrawCircle = False
                else:
                    self._shouldDrawCircle = True
            elif self._currentAdjusting == self.SHOULD_DRAW_TRACKS:
                if  self._shouldDrawTracks:
                    self._shouldDrawTracks = False
                else:
                    self._passedPoints = []  # 軌跡を消去する
                    self._shouldDrawTracks = True
            elif self._currentAdjusting == self.SHOULD_DRAW_DISPLACEMENT_VECTOR:
                if  self._shouldDrawDisplacementVector:
                    self._shouldDrawDisplacementVector = False
                else:
                    self._shouldDrawDisplacementVector = True
            elif self._currentAdjusting == self.SHOULD_DRAW_VEROCITY_VECTOR:
                if  self._shouldDrawVerocityVector:
                    self._shouldDrawVerocityVector = False
                else:
                    self._shouldDrawVerocityVector = True
            elif self._currentAdjusting == self.SHOULD_DRAW_ACCELERATION_VECTOR:
                if  self._shouldDrawAccelerationVector:
                    self._shouldDrawAccelerationVector = False
                else:
                    self._shouldDrawAccelerationVector = True
            elif self._currentAdjusting == self.GRAVITY_STRENGTH:
                pitch = 50  if keycode == 0 else -50
                self._gravityStrength += pitch
            elif self._currentAdjusting == self.SHOULD_DRAW_FORCE_VECTOR:
                if  self._shouldDrawForceVector:
                    self._shouldDrawForceVector = False
                else:
                    self._shouldDrawForceVector = True
            elif self._currentAdjusting == self.SHOULD_DRAW_SYNTHESIZED_VECTOR:
                if  self._shouldDrawSynthesizedVector:
                    self._shouldDrawSynthesizedVector = False
                else:
                    self._shouldDrawSynthesizedVector = True
            elif self._currentAdjusting == self.SHOULD_TRACK_CIRCLE:
                if self._shouldTrackCircle:
                    self._shouldTrackCircle = False
                else:
                    self._shouldTrackCircle = True
            elif self._currentAdjusting == self.SHOULD_DRAW_CANNY_EDGE:
                self._shouldDrawCannyEdge = \
                    not self._shouldDrawCannyEdge
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