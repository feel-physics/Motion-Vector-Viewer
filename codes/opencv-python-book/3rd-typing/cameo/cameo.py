# coding=utf-8
__author__ = 'weed'

import cv2
from datetime import datetime
import numpy

import filters
from managers import WindowManager, CaptureManager
import utils
import my_lib

class Cameo(object):

    ##### TODO: 不要になったオプションは廃止する
    ADJUSTING_OPTIONS = (
        HUE_MIN,
        HUE_MAX,
        VALUE_MIN,
        VALUE_MAX,
        # SHOULD_PROCESS_GAUSSIAN_BLUR,
        # GAUSSIAN_BLUR_KERNEL_SIZE,
        # SHOULD_PROCESS_CLOSING,
        # CLOSING_ITERATIONS,
        # HOUGH_CIRCLE_RESOLUTION,
        # HOUGH_CIRCLE_CANNY_THRESHOLD,
        # HOUGH_CIRCLE_ACCUMULATOR_THRESHOLD,
        # SHOULD_DRAW_CANNY_EDGE,
        # SHOULD_DRAW_CIRCLE,
        SHOULD_DRAW_TRACKS,
        # SHOULD_DRAW_DISPLACEMENT_VECTOR,
        SHOULD_DRAW_VELOCITY_VECTOR,
        SHOULD_DRAW_ACCELERATION_VECTOR,
        IS_MODE_PENDULUM,
        # NUM_FRAMES_DELAY,
        GRAVITY_STRENGTH,
        # SHOULD_PROCESS_QUICK_MOTION,
        SHOULD_DRAW_FORCE_VECTOR_BOTTOM,
        # SHOULD_DRAW_FORCE_VECTOR_TOP,
        CO_FORCE_VECTOR_STRENGTH,
        SHOULD_DRAW_SYNTHESIZED_VECTOR,
        SHOULD_TRACK_CIRCLE,
        SHOULD_DRAW_TRACKS_IN_STROBE_MODE,
        SHOULD_DRAW_VELOCITY_VECTORS_IN_STROBE_MODE,
        SHOWING_FRAME
    ) = range(0, 16)

    SHOWING_FRAME_OPTIONS = (
        ORIGINAL,
        WHAT_COMPUTER_SEE
    ) = range(0, 2)

    def __init__(self):

        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, False)

        ### Filtering
        # self._hueMin                       = 40  # 色紙
        # self._hueMax                       = 60  # 色紙
        self._hueMin                       = 60  # テニスボール
        self._hueMax                       = 80  # テニスボール
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
        self._shouldDrawVelocityVector     = True
        self._shouldDrawAccelerationVector = False
        self._shouldDrawForceVectorBottom  = False
        self._shouldDrawForceVectorTop     = False
        self._gravityStrength              = 200
        self._shouldDrawSynthesizedVector  = False

        self._currentAdjusting             = self.IS_MODE_PENDULUM
        self._currentShowing               = self.ORIGINAL

        self._numFramesDelay               = 6  # 13
        self._enteredFrames                = []
        self._populationVelocity           = 6
        self._populationAcceleration       = 6  # 12
        self._indexQuickMotion             = None
        self._shouldProcessQuickMotion     = False
        self._coForceVectorStrength        = 7.0
        self._isModePendulum               = False

        # ストロボモード 15/08/12 -
        self._shouldDrawTracksInStrobeMode = False
        self._numStrobeModeSkips           = 5
        self._velocityVectorsHistory       = []
        self._shouldDrawVelocityVectorsInStrobeMode = True

        self._timeSelfTimerStarted         = None

        self._corners                      = None

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
        # FPS計算用
        self._fpsWithTick = my_lib.fpsWithTick()

        # ウィンドウが存在する限り・・・
        while self._windowManager.isWindowCreated:
            # フレームを取得し・・・
            self._captureManager.enterFrame()
            frameToDisplay = self._captureManager.frame
            frameToDisplay[:] = numpy.fliplr(frameToDisplay)

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

            densityTrackWindow = -1

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

            if self._currentShowing == self.WHAT_COMPUTER_SEE:
                # pass
                # gray = getMaskToFindCircle(self, frameToDisplay)
                gray = getMaskToFindCircle(self, frameNow)
                cv2.merge((gray, gray, gray), frameToDisplay)

            elif self._currentShowing == self.ORIGINAL:
                pass


            ### 物体追跡・描画処理 ###


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

                # 8近傍
                element8 = numpy.array([[1,1,1],
                                        [1,1,1],
                                        [1,1,1]], numpy.uint8)
                # オープニング
                cv2.morphologyEx(dst, cv2.MORPH_OPEN, element8, dst, None, 2)

                # バックプロジェクションを描画するコード（デバッグ用）
                if self._currentShowing == self.WHAT_COMPUTER_SEE:
                    cv2.merge((dst, dst, dst), frameToDisplay)

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
                        self._passedPoints = []  # 軌跡を消去する
                        self._velocityVectorsHistory = []
                        self._indexQuickMotion = 0
                        # print 'tracking interrupted'

                x,y,w,h = self._track_window

                # 追跡している領域を描く
                if self._currentShowing == self.WHAT_COMPUTER_SEE:
                    cv2.rectangle(frameToDisplay, (x,y), (x+w,y+h),(0,0,200),5)

                # 次の円を検出したら・・・
                if self._track_window is not None:
                    # 通過点リストの最後に要素を追加する
                    self._passedPoints.append((x+w/2, y+h/2))
                    # 速度ベクトルを記録する
                    lastVelocityVector = utils.getVelocityVector(
                        self._passedPoints, self._populationVelocity,
                        self._numFramesDelay
                    )
                    self._velocityVectorsHistory.append(lastVelocityVector)

                # 次の円が見つかっても見つからなくても・・・
                if len(self._passedPoints) - self._numFramesDelay > 0:
                    numPointsVisible = len(self._passedPoints) - self._numFramesDelay

                    # 軌跡を描画する
                    if self._shouldDrawTracks:
                        for i in range(numPointsVisible - 1):
                            cv2.line(frameToDisplay, self._passedPoints[i],
                                     self._passedPoints[i+1], (255,255,255), 5)
                            # # 軌跡ではなく打点する（デバッグ用）
                            # cv2.circle(frameToDisplay, self._passedPoints[i], 1, (255,255,255), 5)

                    # ストロボモード
                    if self._shouldDrawTracksInStrobeMode:
                        for i in range(numPointsVisible - 1):
                            if i % self._numStrobeModeSkips == 0:
                                cv2.circle(frameToDisplay, self._passedPoints[i], 5, (255,255,255), -1)

                    lastPt = self._passedPoints[numPointsVisible-1]

                    # 変位ベクトルを描画する
                    if self._shouldDrawDisplacementVector:
                        vector = (lastPt[0] - self._passedPoints[0][0],
                                  lastPt[1] - self._passedPoints[0][1])
                        if vector is not None:
                            utils.cvArrow(frameToDisplay, self._passedPoints[0],
                                          vector, 1, (255,255,255), 5)

                    # 速度ベクトルを描画する
                    if self._shouldDrawVelocityVector and \
                                    self._velocityVectorsHistory[numPointsVisible - 1] is not None:
                        utils.cvArrow(
                            frameToDisplay, lastPt,
                            self._velocityVectorsHistory[numPointsVisible - 1], 4, (255,0,0), 5)

                    # 速度ベクトルをストロボモードで表示する
                    if self._shouldDrawVelocityVectorsInStrobeMode:
                        for i in range(numPointsVisible - 1):
                            if i % self._numStrobeModeSkips == 0 and \
                                    self._velocityVectorsHistory[i] is not None:
                                utils.cvArrow(
                                    frameToDisplay,
                                    self._passedPoints[i - self._numFramesDelay],
                                    self._velocityVectorsHistory[i],
                                    4, (255,0,0), 5
                                )

                    # 加速度ベクトルを求める
                    # vector = utils.getAccelerationVector(self._passedPoints, self._numFramesDelay*2)
                    # vector = utils.getAccelerationVectorVelocitySensitive(self._passedPoints)
                    aclVector = None
                    if self._shouldProcessQuickMotion:
                        result = utils.getAccelerationVectorStartStop(
                            self._passedPoints, self._populationAcceleration, 3, self._coForceVectorStrength
                        )
                        if result[0] is 'quickMotion':
                            self._indexQuickMotion = len(self._passedPoints)
                            aclVector = result[1]
                        # 急発進／静止後3フレームは通常の加速度を表示しない
                        elif result == 'usual' \
                                and self._indexQuickMotion is not None \
                                and self._indexQuickMotion+4 < numPointsVisible:
                            aclVector = utils.getAccelerationVectorFirFilter(
                                self._passedPoints,
                                self._populationAcceleration,
                                0,
                                self._coForceVectorStrength
                            )
                    else:
                        aclVector = utils.getAccelerationVectorFirFilter(
                            self._passedPoints,
                            self._populationAcceleration,
                            0,
                            self._coForceVectorStrength
                        )

                    # 加速度ベクトルを描画する
                    if self._shouldDrawAccelerationVector:
                        if aclVector is not None:
                            utils.cvArrow(frameToDisplay, lastPt, aclVector, 1, (0,255,0), 5)
                            # print aclVector

                    # 力ベクトルを描画する
                    def drawForceVector(aclVector, ptAcl):
                        if aclVector is None:
                            aclVector = (0,0)
                        vector = (aclVector[0], aclVector[1] - self._gravityStrength)

                        if vector is not None:
                            utils.cvArrow(frameToDisplay, ptAcl, vector, 1, (0,0,255), 5)

                    if self._shouldDrawForceVectorBottom:
                        yPtAcl = self._passedPoints[numPointsVisible-1][1] + h/2
                        ptAcl = (self._passedPoints[numPointsVisible-1][0], yPtAcl)
                        # aclVector = utils.getAccelerationVector(self._passedPoints, self._numFramesDelay*2)
                        drawForceVector(aclVector, ptAcl)

                    if self._shouldDrawForceVectorTop:
                        yPtAcl = self._passedPoints[numPointsVisible-1][1] - h/2
                        ptAcl = (self._passedPoints[numPointsVisible-1][0], yPtAcl)
                        drawForceVector(aclVector, ptAcl)

                    # 力ベクトルの合成を描画する
                    if self._shouldDrawSynthesizedVector:
                        # 手による接触力
                        # aclVector = utils.getAccelerationVector(self._passedPoints, self._numFramesDelay*2)
                        if aclVector is None:
                            aclVector = (0,0)
                        contactForceVector = (aclVector[0], aclVector[1] - self._gravityStrength)
                        if contactForceVector is not None:
                            utils.cvArrow(frameToDisplay, lastPt, contactForceVector, 1, (128,0,255), 2)
                        # 重力
                        gravityForceVector = (0, self._gravityStrength)
                        utils.cvArrow(frameToDisplay, lastPt, gravityForceVector, 1, (0,128,255), 2)
                        # 合力
                        # synthesizedVector = utils.getAccelerationVector(self._passedPoints,
                        #                                                 self._numFramesDelay*2)
                        synthesizedVector = aclVector
                        if synthesizedVector is not None:
                            utils.cvArrow(frameToDisplay, lastPt, synthesizedVector, 1, (0,0,255), 5)
                            # 接触力ベクトルと加速度ベクトルのあいだに線を引く
                            ptSV = (lastPt[0]+synthesizedVector[0], lastPt[1]+synthesizedVector[1])
                            if contactForceVector is not None:
                                ptCF = (lastPt[0]+contactForceVector[0], lastPt[1]+contactForceVector[1])
                                utils.cvLine(frameToDisplay, ptSV, ptCF, (0,0,255), 1)
                            # 重力ベクトルと加速度ベクトルのあいだに線を引く
                            ptGF = (lastPt[0], lastPt[1]+self._gravityStrength)
                            utils.cvLine(frameToDisplay, ptSV, ptGF, (0,0,255), 1)



            # Cannyエッジ検出
            if self._shouldDrawCannyEdge:
                gray = cv2.cvtColor(frameToDisplay, cv2.COLOR_BGR2GRAY)
                edge = cv2.Canny(gray, self._houghCircleParam1/2, self._houghCircleParam1)
                cv2.merge((edge, edge, edge), frameToDisplay)


            ### 情報表示 ###


            if self._isTracking:
                strIsTracking = 'Tracking'
            else:
                strIsTracking = 'Finding'

            # 情報を表示する
            def putText(text, lineNumber):
                cv2.putText(frameToDisplay, text, (100, 50 + 50 * lineNumber),
                            cv2.FONT_HERSHEY_PLAIN, 2.0, (255,255,255), 3)
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
            # elif cur == self.SHOULD_DRAW_TRACKS:
            #     put('Should Draw Tracks'                 , self._shouldDrawTracks)
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
            # elif cur == self.NUM_FRAMES_DELAY:
            #     put('Number of Delay Frames'             , self._numFramesDelay)
            elif cur == self.SHOULD_DRAW_SYNTHESIZED_VECTOR:
                put('Should Draw Synthesized Vector'     , self._shouldDrawSynthesizedVector)
            elif cur == self.SHOULD_TRACK_CIRCLE:
                put('Should Track Circle'                , self._shouldTrackCircle)
            # elif cur == self.SHOULD_DRAW_CANNY_EDGE:
            #     put('Should Draw Canny Edge'             , self._shouldDrawCannyEdge)
            elif cur == self.SHOULD_DRAW_TRACKS_IN_STROBE_MODE:
                put('Should Draw Tracks In Strobe Mode'  , self._shouldDrawTracksInStrobeMode)
            elif cur == self.SHOULD_DRAW_VELOCITY_VECTORS_IN_STROBE_MODE:
                put('Should Draw Velocity Vectors In Strobe Mode' , self._shouldDrawVelocityVectorsInStrobeMode)
            elif cur == self.SHOWING_FRAME:
                if   self._currentShowing == self.ORIGINAL:
                    currentShowing = 'Original'
                # elif self._currentShowing == self.GRAY_SCALE:
                #     currentShowing = 'Gray Scale'
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
            # elif self._currentAdjusting == self.HOUGH_CIRCLE_RESOLUTION:
            #     pitch = 1  if keycode == 0 else -1
            #     self._houghCircleDp     += pitch
            # elif self._currentAdjusting == self.HOUGH_CIRCLE_CANNY_THRESHOLD:
            #     pitch = 20 if keycode == 0 else -20
            #     self._houghCircleParam1 += pitch
            # elif self._currentAdjusting == self.HOUGH_CIRCLE_ACCUMULATOR_THRESHOLD:
            #     pitch = 50 if keycode == 0 else -50
            #     self._houghCircleParam2 += pitch
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
                if  self._shouldDrawTracks:
                    self._shouldDrawTracks = False
                else:
                    self._passedPoints = []  # 軌跡を消去する
                    self._shouldDrawTracks = True
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
            # elif self._currentAdjusting == self.NUM_FRAMES_DELAY:
            #     pitch = 1  if keycode == 0 else -1
            #     self._numFramesDelay += pitch
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
                self._shouldDrawTracksInStrobeMode = \
                    not self._shouldDrawTracksInStrobeMode
            elif self._currentAdjusting == self.SHOULD_DRAW_VELOCITY_VECTORS_IN_STROBE_MODE:
                self._shouldDrawVelocityVectorsInStrobeMode = \
                    not self._shouldDrawVelocityVectorsInStrobeMode
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