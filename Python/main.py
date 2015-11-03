# coding=utf-8
__author__ = 'weed'

import cv2
from datetime import datetime
import numpy
from managers import WindowManager, CaptureManager
from abc import *
import utils

### TODO:このファイルが長すぎる。目的の箇所を探すのが大変。

class Main(object):

    ##### TODO: 不要になったオプションは廃止する
    ADJUSTING_OPTIONS = [
        CAPTURE_BACKGROUND_FRAME,
        SHOULD_DRAW_VELOCITY_VECTORS_VERTICALLY_IN_STROBE_MODE,
        SHOULD_DRAW_TRACKS,
        SHOULD_DRAW_VELOCITY_VECTOR,
        SHOULD_DRAW_VELOCITY_VECTOR_X_COMPONENT,
        SHOULD_DRAW_VELOCITY_VECTORS_X_COMPONENT_IN_STROBE_MODE,
        SHOULD_DRAW_VELOCITY_VECTORS_X_COMPONENT_VERTICALLY_IN_STROBE_MODE,
        NUM_STROBE_SKIPS,
        SPACE_BETWEEN_VERTICAL_VECTORS,
        LENGTH_TIMES_VERTICAL_VELOCITY_VECTORS,

        SHOWING_FRAME,
        HUE_MIN,
        HUE_MAX,
        VALUE_MIN,
        VALUE_MAX
    ] = range(15)

    UNUSED_OPTIONS = [
        DIFF_OF_BACKGROUND_AND_FOREGROUND,
        CO_VELOCITY_VECTOR_STRENGTH,
        SHOULD_DRAW_ACCELERATION_VECTOR,
        IS_MODE_PENDULUM,
        NUM_FRAMES_DELAY,
        GRAVITY_STRENGTH,
        # SHOULD_PROCESS_QUICK_MOTION,
        SHOULD_DRAW_FORCE_VECTOR_BOTTOM,
        # SHOULD_DRAW_FORCE_VECTOR_TOP,
        CO_FORCE_VECTOR_STRENGTH,
        SHOULD_DRAW_SYNTHESIZED_VECTOR,
        SHOULD_TRACK_CIRCLE,
        SHOULD_DRAW_TRACKS_IN_STROBE_MODE,
        HOUGH_CIRCLE_RADIUS_MIN,
        # SHOULD_DRAW_CANNY_EDGE,
        # SHOULD_DRAW_CIRCLE,
        # SHOULD_DRAW_DISPLACEMENT_VECTOR,
        SHOULD_PROCESS_GAUSSIAN_BLUR,
        GAUSSIAN_BLUR_KERNEL_SIZE,
        SHOULD_PROCESS_CLOSING,
        CLOSING_ITERATIONS,
        HOUGH_CIRCLE_RESOLUTION,
        HOUGH_CIRCLE_CANNY_THRESHOLD,
        HOUGH_CIRCLE_ACCUMULATOR_THRESHOLD,
        SHOULD_DRAW_VELOCITY_VECTORS_IN_STROBE_MODE,
    ] = [-1 for x in range(20)]  # すべて-1、合わせて35

    SHOWING_FRAME_OPTIONS = [
        ORIGINAL,
        WHAT_COMPUTER_SEE,
        BEFORE_MASK
    ] = range(3)

    def __init__(self):

        self._scaleRatio                   = 0.6
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, False, self._scaleRatio)

        ### Filtering
        # self._hueMin                       = 40  # 色紙
        # self._hueMax                       = 60  # 色紙
        self._hueMin                       = 30 #マグネット # 50 # テニスボール
        self._hueMax                       = 80 # テニスボール
        self._sThreshold                   = 5
        self._valueMin                     = 130 #220 #60
        self._valueMax                     = 255
        self._gamma                        = 100
        self._shouldProcessGaussianBlur    = True
        self._gaussianBlurKernelSize       = 20
        self._shouldProcessClosing         = True
        self._closingIterations            = 2

        ### Ball Tracking ###
        self._houghCircleDp                = 4
        self._houghCircleParam1            = 100
        self._houghCircleParam2            = 150
        self._houghCircleRadiusMin         = 100
        self._shouldDrawCannyEdge          = False

        self._centerPointOfCircle          = None
        self._positionHistory                 = []

        self._shouldTrackCircle            = True
        self._isTracking                   = False
        self._track_window                 = None
        self._roi_hist                     = None

        self._shouldDrawCircle             = False
        self._shouldDrawTrack              = False
        self._shouldDrawDisplacementVector = False
        self._shouldDrawVelocityVector     = False
        self._shouldDrawAccelerationVector = False
        self._shouldDrawForceVectorBottom  = False
        self._shouldDrawForceVectorTop     = False
        self._gravityStrength              = 200
        self._shouldDrawSynthesizedVector  = False

        self._currentAdjusting             = self.CAPTURE_BACKGROUND_FRAME
        self._currentShowing               = self.ORIGINAL

        self._numFramesDelay               = 6  # 6, 12
        self._enteredFrames                = []
        self._populationVelocity           = 6  # 6, 12
        self._populationAcceleration       = 12  # 12, 24
        self._indexQuickMotion             = None
        self._shouldProcessQuickMotion     = False
        self._coForceVectorStrength        = 7.0
        self._isModePendulum               = False

        # ストロボモード 15/08/12 -
        self._shouldDrawTrackInStrobeMode  = False
        self._numStrobeModeSkips           = 3 # 画面が狭い # 5
        self._velocityVectorsHistory       = []
        self._shouldDrawVelocityVectorsInStrobeMode = False
        self._spaceBetweenVerticalVectors  = 20 # 画面が狭い  # 3
        self._shouldDrawVelocityVectorsVerticallyInStrobeMode = False
        self._shouldDrawVelocityVectorXComponent = False
        self._shouldDrawVelocityVectorsXComponentInStrobeMode = False
        self._coVelocityVectorStrength     = 4
        self._shouldDrawVelocityVectorsXComponentVerticallyInStrobeMode = False
        self._velocityVectorsXComponentHistory = []
        self._colorVelocityVector          = utils.BLUE
        self._colorVelocityVectorXComponent = utils.SKY_BLUE
        self._thicknessVelocityVector      = 5
        self._thicknessVelocityVectorXComponent = 3
        self._lengthTimesVerticalVelocityVectors = 3 # 画面が狭い # 5

        self._timeSelfTimerStarted         = None

        self._corners                      = None

        # 背景差分 15/09/29
        self._isTakingFrameBackground      = None
        self._frameBackground              = None
        self._diffBgFg                     = 50

        # FPS計算用
        self._fpsWithTick                  = utils.fpsWithTick()

        # クラス化 15/10/07
        self._resetKinetics()
        self._velocityGraph                = Graph(self._numFramesDelay,
                                                   self._numStrobeModeSkips,
                                                   self._spaceBetweenVerticalVectors,
                                                   self._lengthTimesVerticalVelocityVectors,
                                                   utils.BLUE, False, 5)

    def _resetKinetics(self):
        self._position                     = Position(self._numFramesDelay, self._numStrobeModeSkips)
        self._velocityVector               = VelocityVector(self._numFramesDelay,
                                                            self._numStrobeModeSkips,
                                                            self._coVelocityVectorStrength,
                                                            self._colorVelocityVector,
                                                            self._thicknessVelocityVector)
        self._velocityXComponentVector     = VelocityVectorXComponent(self._numFramesDelay,
                                                                      self._numStrobeModeSkips,
                                                                      self._coVelocityVectorStrength,
                                                                      self._colorVelocityVector,
                                                                      self._thicknessVelocityVector)
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
            # カメラからフレームを取得し・・・
            self._captureManager.enterFrame()
            frameToDisplay = self._captureManager.frame
            frameToDisplay[:] = numpy.fliplr(frameToDisplay)

            # 数フレームを配列にためて、
            # 新しいフレームを末尾に追加し、
            # 最初のフレームを取り出して表示する。
            frameNow = frameToDisplay.copy()  # 本当の現在のフレーム
            self._enteredFrames.append(frameToDisplay.copy())  # ディープコピーしないと参照を持って行かれる
            frameToDisplay[:] = self._enteredFrames[0]  # ためたフレームの最初のものを処理するフレームとする
            if len(self._enteredFrames) <= self._numFramesDelay:  # 最初はためる
                pass
            else:
                self._enteredFrames.pop(0)  # たまったら最初のものは削除していく

            densityTrackWindow = -1  # 追跡判定用の変数。0.05未満になれば追跡をやめる。


            ### 画面表示 ###


            def getMaskToFindCircle(self, frame):
                """
                後で円を検出するために、検出用フレームに対して色相フィルタやぼかしなどの処理をする。
                SHOWING_WHAT_COMPUTER_SEEのときは、表示用フレームに対しても同じ処理をする。
                """
                mask = utils.getMaskByHsv(frame, self._hueMin, self._hueMax, self._valueMin, self._valueMax,
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
                    self._houghCircleRadiusMin,  # 円の最小半径
                    1)                        # 円の最大半径
                return circles

            if self._isTakingFrameBackground:
                self._frameBackground = frameToDisplay.copy()
                self._isTakingFrameBackground = False

            if self._frameBackground is not None:
                # 円検出はframeNowに対して行われる
                frameNow = utils.getSubtractedFrame(frameNow, self._frameBackground,
                             self._diffBgFg)

            if self._currentShowing == self.WHAT_COMPUTER_SEE:
                gray = getMaskToFindCircle(self, frameNow)
                cv2.merge((gray, gray, gray), frameToDisplay)
            elif self._currentShowing == self.ORIGINAL:
                pass
            elif self._currentShowing == self.BEFORE_MASK:
                if self._frameBackground is not None:
                    frameSubtracted = utils.getSubtractedFrame(frameToDisplay, self._frameBackground,
                                                     self._diffBgFg)
                    gray = getMaskToFindCircle(self, frameSubtracted)
                    frameGray = frameToDisplay.copy()
                    cv2.merge((gray, gray, gray), frameGray)
                    frameToDisplay[:] = cv2.addWeighted(frameSubtracted, 0.3, frameGray, 1.0, 0)
                else:
                    pass



            ### 履歴系描画処理 ###


            # # 表示可能な軌跡があれば・・・
            # if 0 < len(self._positionHistory) - self._numFramesDelay:
            #     numPointsVisible = len(self._positionHistory) - self._numFramesDelay
            #
            #     # 速度ベクトルをストロボモードで描画する
            #     if self._shouldDrawVelocityVectorsInStrobeMode:
            #         utils.drawVelocityVectorsInStrobeMode(
            #             frameToDisplay, self._positionHistory, self._numFramesDelay,
            #             self._numStrobeModeSkips, self._velocityVectorsHistory)
            #     # 速度ベクトルの大きさのグラフを描画する
            #     if self._shouldDrawVelocityVectorsVerticallyInStrobeMode:
            #         utils.drawVelocityVectorsVerticallyInStrobeMode(
            #             frameToDisplay, self._positionHistory,
            #             self._velocityVectorsHistory, self._numFramesDelay,
            #             self._numStrobeModeSkips, self._spaceBetweenVerticalVectors,
            #             self._colorVelocityVector, self._thicknessVelocityVector, True,
            #             self._lengthTimesVerticalVelocityVectors)
            #     # 速度x成分ベクトルをストロボモードで表示する
            #     if self._shouldDrawVelocityVectorsXComponentInStrobeMode:
            #         utils.drawVelocityVectorsInStrobeMode(
            #             frameToDisplay, self._positionHistory, self._numFramesDelay,
            #             self._numStrobeModeSkips, self._velocityVectorsXComponentHistory,
            #             utils.SKY_BLUE, 3)
            #     # 速度ベクトルのx成分（正負あり）のグラフを表示する
            #     if self._shouldDrawVelocityVectorsXComponentVerticallyInStrobeMode:
            #         utils.drawVelocityVectorsVerticallyInStrobeMode(
            #             frameToDisplay, self._positionHistory,
            #             self._velocityVectorsXComponentHistory,
            #             self._numFramesDelay, self._numStrobeModeSkips,
            #             self._spaceBetweenVerticalVectors,
            #             self._colorVelocityVectorXComponent,
            #             self._thicknessVelocityVectorXComponent, True,
            #             self._lengthTimesVerticalVelocityVectors)


            ### 物体追跡 ###


            if self._shouldTrackCircle and not self._isTracking:
                # 検出用フレームをつくる
                frameToFindCircle = getMaskToFindCircle(self, frameNow)
                circles = getCircles(self, frameToFindCircle)  # 円を検出する

                if circles is not None:  # もし円を見つけたら・・・
                    x, y, r = circles[0][0]  # 中心座標と半径を取得して・・・
                    x, y ,r = int(x), int(y), int(r)  # 整数にする
                    # 画面外に円がはみ出す場合は・・・
                    height, width = frameToFindCircle.shape
                    m = 10  # マージン
                    # 画面の中に円が収まる場合
                    if m < x-r \
                            or width < x+2*r+m \
                            or m < y-r \
                            or height < y+2*r+m:

                        ### 新しい円を見つけた！ ###

                        # 追跡したい領域の初期設定
                        self._track_window = (x-r, y-r, 2*r, 2*r)
                        # 追跡のためのROI関心領域（Region of Interest)を設定
                        # print(x, y, r)
                        roi = frameNow[y-r:y+r, x-r:x+r]
                        # HSV色空間に変換
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        if hsv_roi is None:
                            pass
                        else:
                            # マスク画像の作成
                            # 以下の行で「dtype=numpy.uint8」を指定しないと以下のエラーが出る。
                            # > OpenCV Error: Sizes of input arguments do not match
                            # > (The lower boundary is neither
                            # > an array of the same size and same type as src, nor a scalar)
                            # > in inRange
                            mask = cv2.inRange(hsv_roi,
                                    numpy.array([
                                        self._hueMin / 2,           # H最小値
                                        2 ** self._sThreshold - 1,  # S最小値
                                        self._valueMin              # V最小値
                                    ], dtype=numpy.uint8),
                                    # ]),
                                    numpy.array([
                                        self._hueMax / 2,           # H最大値
                                        255,                        # S最大値
                                        self._valueMax              # V最大値
                                    ], dtype=numpy.uint8))
                                    # ]))
                            # ヒストグラムの計算
                            self._roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
                            # ヒストグラムの正規化
                            cv2.normalize(self._roi_hist,self._roi_hist,0,255,cv2.NORM_MINMAX)

                            self._isTracking = True

                            ### 履歴系の初期化 ###

                            # self._positionHistory = []
                            # self._velocityVectorsHistory = []
                            # self._velocityVectorsXComponentHistory = []

                            del self._position
                            del self._velocityVector
                            self._resetKinetics()

            # if self._shouldTrackCircle and not self._isTracking:
            elif self._shouldTrackCircle:  # and self._isTracking:

                dst = utils.getBackProjectFrame(frameNow, self._roi_hist)
                # バックプロジェクションを描画するコード（デバッグ用）
                if self._currentShowing == self.WHAT_COMPUTER_SEE:
                    cv2.merge((dst, dst, dst), frameToDisplay)
                # elif self._currentShowing == self.BEFORE_MASK:
                #     dst_delayed = utils.getBackProjectFrame(frameToDisplay, self._roi_hist)
                #     cv2.merge((dst_delayed, dst_delayed, dst_delayed), frameToDisplay)

                dst = utils.getBackProjectFrame(frameNow, self._roi_hist)

                # 新しい場所を取得するためにmeanshiftを適用
                _, self._track_window = cv2.meanShift(dst, self._track_window,
                                                        ( cv2.TERM_CRITERIA_EPS |
                                                          cv2.TERM_CRITERIA_COUNT, 10, 1 ))

                # 追跡中のウィンドウの密度を計算する
                x, y, w, h = self._track_window
                densityTrackWindow = cv2.mean(dst[y:y+h, x:x+w])[0] / 256

                # 密度が0.05未満なら追跡を中断する
                if densityTrackWindow < 0.05:
                        self._isTracking = False
                        # self._positionHistory = []  # 軌跡を消去する
                        # self._velocityVectorsHistory = []
                        self._indexQuickMotion = 0
                        # print 'tracking interrupted'

                x,y,w,h = self._track_window

                # 追跡している領域を描く
                if self._currentShowing == self.WHAT_COMPUTER_SEE:
                    cv2.rectangle(frameToDisplay, (x,y), (x+w,y+h),(0,0,200),5)

                # 次の円を検出したら・・・
                if self._track_window is not None:


                    ### 位置、速度ベクトル、速度x成分ベクトルを求める ###
                    ### 履歴系に新たに追加 ###


                    # 通過点リストの最後に要素を追加する
                    self._position.addNewVector((x+w/2, y+h/2))
                    self._positionHistory.append((x+w/2, y+h/2))
                    # # 速度ベクトルを記録する
                    # lastVelocityVector = utils.getVelocityVector(
                    #     self._positionHistory, self._populationVelocity,
                    #     self._numFramesDelay
                    # )

                    # 速度ベクトルを記録する
                    lastVelocityVector = utils.getVelocityVector(
                        self._position.history, self._populationVelocity,
                        self._numFramesDelay
                    )
                    self._velocityVector.addNewVector(lastVelocityVector)

                    # self._velocityVectorsHistory.append(lastVelocityVector)
                    # # 速度x成分ベクトルを記録する
                    # lastVelocityVectorXComponent = \
                    #     utils.getComponentVector(lastVelocityVector, "x")
                    # self._velocityVectorsXComponentHistory.append(
                    #     lastVelocityVectorXComponent
                    # )

                    # 速度x成分ベクトルを記録する
                    self._velocityXComponentVector.addNewVector(lastVelocityVector)



                ### 描画処理 ###


                # 次の円が見つかっても見つからなくても・・・
                if len(self._positionHistory) - self._numFramesDelay > 0:
                    numPointsVisible = len(self._positionHistory) - self._numFramesDelay
                    lastPosition = self._positionHistory[numPointsVisible-1]

                    # # 速度ベクトル関係
                    # if self._velocityVectorsHistory[numPointsVisible - 1] is not None:
                    #     c = self._coVelocityVectorStrength
                    #     # 速度x成分ベクトルを描く
                    #     if self._shouldDrawVelocityVectorXComponent:
                    #         v  = self._velocityVectorsHistory[numPointsVisible - 1]
                    #         # 成分ベクトルを求める
                    #         vx = utils.getComponentVector(v, "x")
                    #         utils.cvArrow(
                    #             frameToDisplay, lastPosition, vx, c,
                    #             self._colorVelocityVectorXComponent,
                    #             self._thicknessVelocityVectorXComponent)
                    #
                    #         # 速度ベクトルと速度x成分ベクトルの両方が表示しているときは
                    #         # 2つのベクトルの先を結ぶ線分を描く
                    #         if self._shouldDrawVelocityVector:
                    #             # 元ベクトルの先から成分ベクトルの先へ線を引く
                    #             utils.cvLine(frameToDisplay,
                    #                          (lastPosition[0] + v[0]*c, lastPosition[1] + v[1]*c),
                    #                          (lastPosition[0] + v[0]*c, lastPosition[1]),
                    #                          utils.WHITE, 2)

                    # 加速度ベクトルを求める
                    # vector = utils.getAccelerationVector(self._passedPoints, self._numFramesDelay*2)
                    # vector = utils.getAccelerationVectorVelocitySensitive(self._passedPoints)
                    aclVector = None
                    if self._shouldProcessQuickMotion:
                        result = utils.getAccelerationVectorStartStop(
                            self._positionHistory, self._populationAcceleration, 3, self._coForceVectorStrength
                        )
                        if result[0] is 'quickMotion':
                            self._indexQuickMotion = len(self._positionHistory)
                            aclVector = result[1]
                        # 急発進／静止後3フレームは通常の加速度を表示しない
                        elif result == 'usual' \
                                and self._indexQuickMotion is not None \
                                and self._indexQuickMotion+4 < numPointsVisible:
                            aclVector = utils.getAccelerationVectorFirFilter(
                                self._positionHistory,
                                self._populationAcceleration,
                                0,
                                self._coForceVectorStrength
                            )
                    else:
                        # aclVector = utils.getAccelerationVectorFirFilter(
                        #     self._positionHistory,
                        #     self._populationAcceleration,
                        #     0,
                        #     self._coForceVectorStrength
                        # )
                        aclVector = utils.getAccelerationVector2(
                            self._positionHistory, self._populationVelocity,
                            self._populationAcceleration)

                    # 加速度ベクトルを描画する
                    if self._shouldDrawAccelerationVector:
                        if aclVector is not None:
                            utils.cvArrow(frameToDisplay, lastPosition, aclVector, 1, utils.GREEN, 5)
                            # print aclVector

                    # 力ベクトルを物体の底面を基点として描画する
                    if self._shouldDrawForceVectorBottom:
                        yPositionAclBegin = \
                            self._positionHistory[numPointsVisible-1][1] + h/2
                        positionAclBegin  = (self._positionHistory[numPointsVisible-1][0],
                                        yPositionAclBegin)
                        # aclVector = utils.getAccelerationVector(self._passedPoints, self._numFramesDelay*2)
                        utils.drawForceVector(frameToDisplay, aclVector,
                                              positionAclBegin, self._gravityStrength)

                    # 力ベクトルを物体のてっぺんを基点として描画する
                    if self._shouldDrawForceVectorTop:
                        yPositionAclBegin = \
                            self._positionHistory[numPointsVisible-1][1] - h/2
                        positionAclBegin  = (self._positionHistory[numPointsVisible-1][0],
                                        yPositionAclBegin)
                        utils.drawForceVector(frameToDisplay, aclVector,
                                              positionAclBegin, self._gravityStrength)

                    # 力ベクトルの合成を描画する
                    if self._shouldDrawSynthesizedVector:
                        # 手による接触力
                        # aclVector = utils.getAccelerationVector(self._passedPoints, self._numFramesDelay*2)
                        if aclVector is None:
                            aclVector = (0,0)
                        contactForceVector = \
                            (aclVector[0], aclVector[1] - self._gravityStrength)
                        if contactForceVector is not None:
                            utils.cvArrow(frameToDisplay, lastPosition,
                                          contactForceVector, 1, (128,0,255), 2)
                        # 重力
                        gravityForceVector = (0, self._gravityStrength)
                        utils.cvArrow(frameToDisplay, lastPosition,
                                      gravityForceVector, 1, (0,128,255), 2)
                        # 合力
                        # synthesizedVector = utils.getAccelerationVector(self._passedPoints,
                        #                                                 self._numFramesDelay*2)
                        synthesizedVector = aclVector
                        if synthesizedVector is not None:
                            utils.cvArrow(
                                frameToDisplay, lastPosition, synthesizedVector,
                                1, utils.BLUE, 5
                            )
                            # 接触力ベクトルと加速度ベクトルのあいだに線を引く
                            positionSVBegin = \
                                (lastPosition[0]+synthesizedVector[0],
                                 lastPosition[1]+synthesizedVector[1])
                            if contactForceVector is not None:
                                positionCFBegin = \
                                    (lastPosition[0]+contactForceVector[0],
                                     lastPosition[1]+contactForceVector[1])
                                utils.cvLine(frameToDisplay, positionSVBegin,
                                             positionCFBegin, utils.BLUE, 1)
                            # 重力ベクトルと加速度ベクトルのあいだに線を引く
                            positionGFBegin = \
                                (lastPosition[0], lastPosition[1]+self._gravityStrength)
                            utils.cvLine(frameToDisplay, positionSVBegin,
                                         positionGFBegin, utils.BLUE, 1)


            ### 描画処理 ###


            indexJustDisplaying = len(self._position.history) - self._numFramesDelay - 1
            if 0 <= indexJustDisplaying:
                # 軌跡をストロボモードで描画する
                if self._shouldDrawTrackInStrobeMode:
                    self._position.drawInStrobeMode(frameToDisplay)
                # 軌跡を描画する
                if self._shouldDrawTrack:
                    self._position.drawTrack(frameToDisplay)
                    # # 軌跡ではなく打点する（デバッグ用）
                    # self._position.draw(frameToDisplay)
                    # cv2.circle(frameToDisplay, self._passedPoints[i], 1, WHITE, 5)
                # 変位ベクトルを描画する
                if self._shouldDrawDisplacementVector:
                    self._position.drawDisplacementVector(frameToDisplay)
                # 速度ベクトルを描く
                if self._shouldDrawVelocityVector:
                    self._velocityVector.draw(frameToDisplay, self._position)
                # 速度ベクトルをストロボモードで表示する
                if self._shouldDrawVelocityVectorsInStrobeMode:
                    self._velocityVector.drawInStrobeMode(frameToDisplay, self._position)
                # 速度グラフを描く
                if self._shouldDrawVelocityVectorsVerticallyInStrobeMode:
                    self._velocityGraph.draw(frameToDisplay, self._velocityVector)
                # 速度x成分ベクトルを描く
                if self._shouldDrawVelocityVectorXComponent:
                    self._velocityXComponentVector.draw(frameToDisplay, self._position)
                    #         v  = self._velocityVectorsHistory[numPointsVisible - 1]
                    #         # 成分ベクトルを求める
                    #         vx = utils.getComponentVector(v, "x")
                    #         utils.cvArrow(
                    #             frameToDisplay, lastPosition, vx, c,
                    #             self._colorVelocityVectorXComponent,
                    #             self._thicknessVelocityVectorXComponent)
                    #
                    #         # 速度ベクトルと速度x成分ベクトルの両方が表示しているときは
                    #         # 2つのベクトルの先を結ぶ線分を描く
                    #         if self._shouldDrawVelocityVector:
                    #             # 元ベクトルの先から成分ベクトルの先へ線を引く
                    #             utils.cvLine(frameToDisplay,
                    #                          (lastPosition[0] + v[0]*c, lastPosition[1] + v[1]*c),
                    #                          (lastPosition[0] + v[0]*c, lastPosition[1]),
                    #                          utils.WHITE, 2)



            ### 画面左上にテキストで情報表示 ###


            if self._isTracking:
                strIsTracking = 'Tracking'
            else:
                strIsTracking = 'Finding'

            # 情報を表示する
            def putText(text, lineNumber):
                cv2.putText(frameToDisplay, text, (50, 20 + 30 * lineNumber),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, utils.WHITE, 2)
            def put(label, value):
                fps = self._fpsWithTick.get()  # FPSを計算する
                putText('FPS '+str(fps)
                        +' '+strIsTracking
                        +' '+"{0:.2f}".format(densityTrackWindow), 1)
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
            # elif cur == self.HOUGH_CIRCLE_RESOLUTION:
            #     put('Hough Circle Resolution'            , self._houghCircleDp)
            # elif cur == self.HOUGH_CIRCLE_CANNY_THRESHOLD:
            #     put('Hough Circle Canny Threshold'       , self._houghCircleParam1)
            # elif cur == self.HOUGH_CIRCLE_ACCUMULATOR_THRESHOLD:
            #     put('Hough Circle Accumulator Threshold' , self._houghCircleParam2)
            elif cur == self.HOUGH_CIRCLE_RADIUS_MIN:
                put('Hough Circle Radius Min'              , self._houghCircleRadiusMin)
            # elif cur == self.GAUSSIAN_BLUR_KERNEL_SIZE:
            #     put('Gaussian Blur Kernel Size'          , self._gaussianBlurKernelSize)
            # elif cur == self.SHOULD_PROCESS_GAUSSIAN_BLUR:
            #     put('Process Gaussian Blur'              , self._shouldProcessGaussianBlur)
            # elif cur == self.SHOULD_PROCESS_CLOSING:
            #     put('Process Closing'                    , self._shouldProcessClosing)
            # elif cur == self.CLOSING_ITERATIONS:
            #     put('Closing Iterations'                 , self._closingIterations)
            # elif cur == self.SHOULD_DRAW_CIRCLE:
            #     put('Should Draw Circle'                 , self._shouldDrawCircle)
            elif cur == self.SHOULD_DRAW_TRACKS:
                put('Should Draw Tracks'                 , self._shouldDrawTrack)
            # elif cur == self.SHOULD_DRAW_DISPLACEMENT_VECTOR:
            #     put('Should Draw Displacement Vector'    , self._shouldDrawDisplacementVector)
            elif cur == self.SHOULD_DRAW_VELOCITY_VECTOR:
                put('Should Draw Velocity Vector'        , self._shouldDrawVelocityVector)
            elif cur == self.SHOULD_DRAW_ACCELERATION_VECTOR:
                put('Should Draw Acceleration Vector'    , self._shouldDrawAccelerationVector)
            elif cur == self.GRAVITY_STRENGTH:
                put('Gravity Strength'                   , self._gravityStrength)
            # elif cur == self.SHOULD_PROCESS_QUICK_MOTION:
            #     put('Should Process Quick Motion'        , self._shouldProcessQuickMotion)
            elif cur == self.SHOULD_DRAW_FORCE_VECTOR_BOTTOM:
                put('Should Draw Force Vector Bottom'    , self._shouldDrawForceVectorBottom)
            # elif cur == self.SHOULD_DRAW_FORCE_VECTOR_TOP:
            #     put('Should Draw Force Vector Top'       , self._shouldDrawForceVectorTop)
            elif cur == self.CO_FORCE_VECTOR_STRENGTH:
                put('Coefficient of Force Vector Strength',self._coForceVectorStrength)
            elif cur == self.IS_MODE_PENDULUM:
                put('Pendulum Mode'                      , self._isModePendulum)
            elif cur == self.NUM_FRAMES_DELAY:
                put('Number of Delay Frames'             , self._numFramesDelay)
            elif cur == self.SHOULD_DRAW_SYNTHESIZED_VECTOR:
                put('Should Draw Synthesized Vector'     , self._shouldDrawSynthesizedVector)
            elif cur == self.SHOULD_TRACK_CIRCLE:
                put('Should Track Circle'                , self._shouldTrackCircle)
            # elif cur == self.SHOULD_DRAW_CANNY_EDGE:
            #     put('Should Draw Canny Edge'             , self._shouldDrawCannyEdge)
            elif cur == self.SHOULD_DRAW_TRACKS_IN_STROBE_MODE:
                put('Should Draw Tracks In Strobe Mode'  , self._shouldDrawTrackInStrobeMode)
            elif cur == self.SHOULD_DRAW_VELOCITY_VECTORS_IN_STROBE_MODE:
                put('Should Draw Velocity Vectors In Strobe Mode' ,
                    self._shouldDrawVelocityVectorsInStrobeMode)
            elif cur == self.SHOULD_DRAW_VELOCITY_VECTORS_VERTICALLY_IN_STROBE_MODE:
                put('Should Draw Velocity Vectors Vertically In Strobe Mode' ,
                    self._shouldDrawVelocityVectorsVerticallyInStrobeMode)
            elif cur == self.SHOULD_DRAW_VELOCITY_VECTOR_X_COMPONENT:
                put('Should Draw Velocity Vector X Component' ,
                    self._shouldDrawVelocityVectorXComponent)
            elif cur == self.CO_VELOCITY_VECTOR_STRENGTH:
                put('Coefficient of Velocity Vector Strength' ,
                    self._coVelocityVectorStrength)
            elif cur == self.SHOULD_DRAW_VELOCITY_VECTORS_X_COMPONENT_IN_STROBE_MODE:
                put('Should Draw Velocity Vectors X Component In Strobe Mode' ,
                    self._shouldDrawVelocityVectorsXComponentInStrobeMode)
            elif cur == self.SHOULD_DRAW_VELOCITY_VECTORS_X_COMPONENT_VERTICALLY_IN_STROBE_MODE:
                put('Should Draw Velocity Vectors X Component Vertically In Strobe Mode' ,
                    self._shouldDrawVelocityVectorsXComponentVerticallyInStrobeMode)
            elif cur == self.CAPTURE_BACKGROUND_FRAME:
                put('Capture Background Frame' ,
                    self._frameBackground is not None)
            elif cur == self.NUM_STROBE_SKIPS:
                put('Number of Skip Frames in Strobe' ,
                    self._numStrobeModeSkips)
            elif cur == self.SPACE_BETWEEN_VERTICAL_VECTORS:
                put('Space between Vertical Vectors' ,
                    self._spaceBetweenVerticalVectors)
            elif cur == self.LENGTH_TIMES_VERTICAL_VELOCITY_VECTORS:
                put('Length Times of Vertical Velocity Vectors' ,
                    self._lengthTimesVerticalVelocityVectors)
            elif cur == self.DIFF_OF_BACKGROUND_AND_FOREGROUND:
                put('Diff of Background and Foreground'       , self._diffBgFg)
            elif cur == self.SHOWING_FRAME:
                if   self._currentShowing == self.ORIGINAL:
                    currentShowing = 'Original'
                elif self._currentShowing == self.WHAT_COMPUTER_SEE:
                    currentShowing = 'What Computer See'
                elif self._currentShowing == self.BEFORE_MASK:
                    currentShowing = 'Before Mask'
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
                    datetime.now().strftime('%Y-%m-%d %H%M%S ')
                    + 'screencast.avi')
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
            # elif self._currentAdjusting == self.HOUGH_CIRCLE_RESOLUTION:
            #     pitch = 1  if keycode == 0 else -1
            #     self._houghCircleDp     += pitch
            # elif self._currentAdjusting == self.HOUGH_CIRCLE_CANNY_THRESHOLD:
            #     pitch = 20 if keycode == 0 else -20
            #     self._houghCircleParam1 += pitch
            # elif self._currentAdjusting == self.HOUGH_CIRCLE_ACCUMULATOR_THRESHOLD:
            #     pitch = 50 if keycode == 0 else -50
            #     self._houghCircleParam2 += pitch
            elif self._currentAdjusting == self.HOUGH_CIRCLE_RADIUS_MIN:
                pitch = 10 if keycode == 0 else -10
                self._houghCircleRadiusMin += pitch
            # elif self._currentAdjusting == self.GAUSSIAN_BLUR_KERNEL_SIZE:
            #     pitch = 1  if keycode == 0 else -1
            #     self._gaussianBlurKernelSize += pitch
            # elif self._currentAdjusting == self.SHOULD_PROCESS_GAUSSIAN_BLUR:
            #     self._shouldProcessGaussianBlur = \
            #         not self._shouldProcessGaussianBlur
            # elif self._currentAdjusting == self.SHOULD_PROCESS_CLOSING:
            #     self._shouldProcessClosing = \
            #         not self._shouldProcessClosing
            # elif self._currentAdjusting == self.CLOSING_ITERATIONS:
            #     pitch = 1  if keycode == 0 else -1
            #     self._closingIterations += pitch
            # elif self._currentAdjusting == self.SHOULD_DRAW_CIRCLE:
            #     if  self._shouldDrawCircle:
            #         self._shouldDrawCircle = False
            #     else:
            #         self._shouldDrawCircle = True
            elif self._currentAdjusting == self.SHOULD_DRAW_TRACKS:
                if  self._shouldDrawTrack:
                    self._shouldDrawTrack = False
                else:
                    self._positionHistory = []  # 軌跡を消去する
                    self._shouldDrawTrack = True
            # elif self._currentAdjusting == self.SHOULD_DRAW_DISPLACEMENT_VECTOR:
            #     if  self._shouldDrawDisplacementVector:
            #         self._shouldDrawDisplacementVector = False
            #     else:
            #         self._shouldDrawDisplacementVector = True
            elif self._currentAdjusting == self.SHOULD_DRAW_VELOCITY_VECTOR:
                if  self._shouldDrawVelocityVector:
                    self._shouldDrawVelocityVector = False
                else:
                    self._shouldDrawVelocityVector = True
            elif self._currentAdjusting == self.SHOULD_DRAW_ACCELERATION_VECTOR:
                if  self._shouldDrawAccelerationVector:
                    self._shouldDrawAccelerationVector = False
                else:
                    self._shouldDrawAccelerationVector = True
            elif self._currentAdjusting == self.GRAVITY_STRENGTH:
                pitch = 100  if keycode == 0 else -100
                self._gravityStrength += pitch
            elif self._currentAdjusting == self.CO_FORCE_VECTOR_STRENGTH:
                pitch = 1.0  if keycode == 0 else -1.0
                self._coForceVectorStrength += pitch
            elif self._currentAdjusting == self.NUM_FRAMES_DELAY:
                pitch = 1  if keycode == 0 else -1
                self._numFramesDelay += pitch
            elif self._currentAdjusting == self.IS_MODE_PENDULUM:
                if self._isModePendulum:
                    self._shouldDrawDisplacementVector = False
                    self._shouldDrawVelocityVector     = False
                    self._shouldDrawAccelerationVector = True
                    self._shouldDrawForceVectorBottom  = False
                    self._shouldDrawForceVectorTop     = False
                    self._shouldDrawSynthesizedVector  = False
                    self._coForceVectorStrength        = 50.0
                    self._shouldProcessQuickMotion     = False
                    self._isModePendulum = False
                else:
                    self._shouldDrawDisplacementVector = False
                    self._shouldDrawVelocityVector     = False
                    self._shouldDrawAccelerationVector = False
                    self._shouldDrawForceVectorBottom  = False
                    self._shouldDrawForceVectorTop     = False
                    self._shouldDrawSynthesizedVector  = True
                    self._gravityStrength              = 200
                    self._coForceVectorStrength        = 13.0
                    self._shouldProcessQuickMotion     = False
                    self._isModePendulum = True
            # elif self._currentAdjusting == self.SHOULD_PROCESS_QUICK_MOTION:
            #     self._shouldProcessQuickMotion = \
            #         not self._shouldProcessQuickMotion
            elif self._currentAdjusting == self.SHOULD_DRAW_FORCE_VECTOR_BOTTOM:
                if  self._shouldDrawForceVectorBottom:
                    self._shouldDrawForceVectorBottom = False
                else:
                    self._shouldDrawForceVectorBottom = True
            # elif self._currentAdjusting == self.SHOULD_DRAW_FORCE_VECTOR_TOP:
            #     self._shouldDrawForceVectorTop = not \
            #         self._shouldDrawForceVectorTop
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
            # elif self._currentAdjusting == self.SHOULD_DRAW_CANNY_EDGE:
            #     self._shouldDrawCannyEdge = \
            #         not self._shouldDrawCannyEdge
            elif self._currentAdjusting == self.SHOULD_DRAW_TRACKS_IN_STROBE_MODE:
                self._shouldDrawTrackInStrobeMode = \
                    not self._shouldDrawTrackInStrobeMode
            elif self._currentAdjusting == self.SHOULD_DRAW_VELOCITY_VECTORS_IN_STROBE_MODE:
                self._shouldDrawVelocityVectorsInStrobeMode = \
                    not self._shouldDrawVelocityVectorsInStrobeMode
                self._positionHistory = []
                self._velocityVectorsHistory = []
            elif self._currentAdjusting == self.SHOULD_DRAW_VELOCITY_VECTORS_VERTICALLY_IN_STROBE_MODE:
                self._shouldDrawVelocityVectorsVerticallyInStrobeMode = \
                    not self._shouldDrawVelocityVectorsVerticallyInStrobeMode
                self._positionHistory = []
                self._velocityVectorsHistory = []
            elif self._currentAdjusting == self.SHOULD_DRAW_VELOCITY_VECTOR_X_COMPONENT:
                self._shouldDrawVelocityVectorXComponent = \
                    not self._shouldDrawVelocityVectorXComponent
            elif self._currentAdjusting == self.CO_VELOCITY_VECTOR_STRENGTH:
                pitch = 1  if keycode == 0 else -1
                self._coVelocityVectorStrength += pitch
            elif self._currentAdjusting == \
                    self.SHOULD_DRAW_VELOCITY_VECTORS_X_COMPONENT_IN_STROBE_MODE:
                self._shouldDrawVelocityVectorsXComponentInStrobeMode = \
                    not self._shouldDrawVelocityVectorsXComponentInStrobeMode
                self._positionHistory = []
                self._velocityVectorsHistory = []
            elif self._currentAdjusting == \
                    self.SHOULD_DRAW_VELOCITY_VECTORS_X_COMPONENT_VERTICALLY_IN_STROBE_MODE:
                self._shouldDrawVelocityVectorsXComponentVerticallyInStrobeMode = \
                    not self._shouldDrawVelocityVectorsXComponentVerticallyInStrobeMode
                self._positionHistory = []
                self._velocityVectorsHistory = []
            elif self._currentAdjusting == self.CAPTURE_BACKGROUND_FRAME:
                self._isTakingFrameBackground = True
            elif self._currentAdjusting == self.DIFF_OF_BACKGROUND_AND_FOREGROUND:
                pitch = 10 if keycode == 0 else -10
                self._diffBgFg += pitch
            elif self._currentAdjusting == self.NUM_STROBE_SKIPS:
                pitch = 1 if keycode == 0 else -1
                self._numStrobeModeSkips += pitch
            elif self._currentAdjusting == self.SPACE_BETWEEN_VERTICAL_VECTORS:
                pitch = 1 if keycode == 0 else -1
                self._spaceBetweenVerticalVectors += pitch
            elif self._currentAdjusting == self.LENGTH_TIMES_VERTICAL_VELOCITY_VECTORS:
                pitch = 1 if keycode == 0 else -1
                self._lengthTimesVerticalVelocityVectors += pitch
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


### クラス設計方針：historyを隠蔽する ###


class BaseVector(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def __init__(self, numFramesDelay, numStrobeModeSkips):
        self.history             = []
        self._numFramesDelay     = numFramesDelay
        self._numStrobeModeSkips = numStrobeModeSkips
    @abstractmethod
    def addNewVector(self, vector):
        self.history.append(vector)

class VectorWithPosition(BaseVector):
    __metaclass__ = ABCMeta
    @abstractmethod
    def _drawWithIndex(self, frame, position, i):
        pass
    @abstractmethod
    def draw(self, frame, position):
        pass
    @abstractmethod
    def drawInStrobeMode(self, frame, position):
        pass

class Position(BaseVector):
    def __init__(self, numFramesDelay, numStrobeModeSkips):
        super(Position, self).__init__(numFramesDelay, numStrobeModeSkips)
    def addNewVector(self, vector):
        super(Position, self).addNewVector(vector)
    def draw(self, frame):
        """
        追跡している点を描画する
        :param frame: 背景フレーム
        :return:
        """
        indexJustDisplaying = len(self.history) - self._numFramesDelay - 1
        if indexJustDisplaying >= 0:
            cv2.circle(frame, self.history[indexJustDisplaying], 1, utils.WHITE, 5)
    def drawInStrobeMode(self, frame):
        """
        追跡している点をストロボ描画する
        :param frame: 背景フレーム
        :return:
        """
        indexJustDisplaying = len(self.history) - self._numFramesDelay - 1
        if 0 <= indexJustDisplaying:
            for i in range(indexJustDisplaying):
                if i % self._numStrobeModeSkips == 0:
                    cv2.circle(frame, self.history[i], 5, utils.WHITE, -1)
    def drawTrack(self, frame):
        """
        軌跡を描画する
        :param frame: 背景フレーム
        :return:
        """
        indexJustDisplaying = len(self.history) - self._numFramesDelay - 1
        if 0 <= indexJustDisplaying:
            for i in range(indexJustDisplaying):
                cv2.line(frame, self.history[i], self.history[i+1], utils.WHITE, 5)
    # 変位ベクトルを描画する
    def drawDisplacementVector(self, frame):
        indexJustDisplaying = len(self.history) - self._numFramesDelay - 1
        if 0 <= indexJustDisplaying:
            vector = (self.history[indexJustDisplaying][0] - self.history[0][0],
                      self.history[indexJustDisplaying][1] - self.history[0][1])
            if vector is not None:
                utils.cvArrow(frame, self.history[0], vector, 1, utils.WHITE, 5)

class VelocityVector(VectorWithPosition):
    def __init__(self, numFramesDelay, numStrobeModeSkips, coVelocityVectorStrength,
                 colorVelocityVector, thicknessVelocityVector):
        super(VelocityVector, self).__init__(numFramesDelay, numStrobeModeSkips)
        self._coVelocityVectorStrength = coVelocityVectorStrength
        self._colorVelocityVector = colorVelocityVector
        self._thicknessVelocityVector = thicknessVelocityVector
    def addNewVector(self, vector):
        super(VelocityVector, self).addNewVector(vector)
    def _drawWithIndex(self, frame, position, i):
        """
        速度ベクトルを描画する
        :param frame: 背景フレーム
        :param Position: 描画位置
        :param i: i番目の速度ベクトルを描画する
        :return:
        """
        utils.cvArrow(
            frame, position.history[i], self.history[i],
            self._coVelocityVectorStrength, self._colorVelocityVector,
            self._thicknessVelocityVector)
    def draw(self, frame, position):
        """
        最新の速度ベクトルを描画する
        :param frame: 背景フレーム
        :param Position: 描画位置
        :return:
        """
        indexJustDisplaying = len(self.history) - self._numFramesDelay - 1
        if 0 <= indexJustDisplaying \
                and self.history[indexJustDisplaying] is not None:
            self._drawWithIndex(frame, position, indexJustDisplaying)
    def drawInStrobeMode(self, frame, position):
        """
        速度ベクトルをストロボ描画する
        :param frame: 背景フレーム
        :param Position: 描画位置
        :return:
        """
        indexJustDisplaying = len(self.history) - self._numFramesDelay - 1
        if 0 <= indexJustDisplaying:
            for i in range(indexJustDisplaying):
                if i % self._numStrobeModeSkips == 0 and self.history[i] is not None:
                    self._drawWithIndex(frame, position, i)

class VelocityVectorXComponent(VelocityVector):
    def addNewVector(self, velocityVector):
        # 成分ベクトルを求める
        velocityXComponentVector = utils.getComponentVector(velocityVector, "x")
        # 追加する
        super(VelocityVectorXComponent, self).addNewVector(velocityXComponentVector)

class Graph(object):
    def __init__(self, numFramesDelay, numStrobeModeSkips, spaceBetweenVerticalVectors,
                 lengthTimes, color, isSigned, thickness):
        """
        コンストラクタ
        :param numFramesDelay: int
        :param numStrobeModeSkips: int
        :param spaceBetweenVerticalVectors: int
        :param lengthTimes: int
        :param color: tuple
        :param isSigned: bool
        :param thickness: int
        :return:
        """
        self._numFramesDelay              = numFramesDelay
        self._numStrobeModeSkips          = numStrobeModeSkips
        self._spaceBetweenVerticalVectors = spaceBetweenVerticalVectors
        self._lengthTimes                 = lengthTimes
        self._color                       = color
        self._isSigned                    = isSigned
        self._thickness                   = thickness
    def draw(self, frame, vector):
        """
        グラフを描画する
        :param frame: 背景フレーム
        :param vector: VectorWithPosition
        :return:
        """
        for i in range(len(vector.history) - self._numFramesDelay - 1):
            if i % self._numStrobeModeSkips == 0 and \
                            vector.history[i]  is not None:
                utils.cvVerticalArrow(
                    frame, self._spaceBetweenVerticalVectors*i/self._numStrobeModeSkips,
                    vector.history[i], self._lengthTimes, self._color,
                    self._isSigned, self._thickness)

if __name__ == "__main__":
    Main().run()