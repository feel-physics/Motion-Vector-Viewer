# coding=utf-8
__author__ = 'weed'

import cv2
import numpy
import math

WHITE    = (255,255,255)
RED      = (  0,  0,255)
GREEN    = (  0,128,  0)
BLUE     = (255,  0,  0)
SKY_BLUE = (255,128,128)

def getVelocityVector(positionHistory, population=1, numFramesDelay=0):
    # populationは母集団。すなわち、何フレーム分の位置データを用いて速度を求めるか。
    # populationが4、numFramesDelayが6の場合は
    # vはPt[-1-6-2=-9],Pt[-1-6+2=-5]を参照する。

    indexPtBegin = -1-numFramesDelay-int(population/2)  # ptBegin: 始点
    indexPtEnd   = -1-numFramesDelay+int(population/2)  # ptEnd  : 終点

    # 追跡開始直後
    if len(positionHistory) < -indexPtBegin \
            or positionHistory[indexPtBegin] is None \
            or positionHistory[indexPtEnd]   is None:
        return None
    else:
        ptBeginNp = numpy.array(positionHistory[indexPtBegin])
        ptEndNp   = numpy.array(positionHistory[indexPtEnd]  )
        # 移動ベクトル Δpt = ptEnd - ptBegin
        deltaPtNp = ptEndNp - ptBeginNp

        # 移動してなければNoneを返す
        notMoved = (deltaPtNp == numpy.array([0,0]))
        if notMoved.all():
            return None
        # 移動していれば、速度ベクトル = 移動ベクトル / 母数
        else:
            velocityVectorNp = deltaPtNp / float(population)
            velocityVector   = tuple(velocityVectorNp)
            return velocityVector

def getAccelerationVector2(velocityVectorsHistory, population=1, numFramesDelay=0,
                           coAcceleration=25):
    # populationは母集団。すなわち、何フレーム分の位置データを用いて加速度を求めるか。
    # populationが4、numFramesDelayが6の場合は
    # aはv[-1-3-2=-6],v[-1-3+2=-2]を参照する。

    # ToDo: なぜ以下の式でnumFramesDelayを半分にするのか、わからない
    # velocityVectorBegin: 始点 -1-3-3
    indexVelocityVectorBegin = -1-int(numFramesDelay/2)-int(population/2)
    # velocityVectorEnd  : 終点 -1-3+3
    indexVelocityVectorEnd   = -1-int(numFramesDelay/2)+int(population/2)

    # 追跡開始直後
    if len(velocityVectorsHistory) < -indexVelocityVectorBegin \
            or velocityVectorsHistory[indexVelocityVectorBegin] is None \
            or velocityVectorsHistory[indexVelocityVectorEnd]   is None:
        return None
    else:
        velocityVectorBeginNp = numpy.array(velocityVectorsHistory[indexVelocityVectorBegin])
        velocityVectorEndNp   = numpy.array(velocityVectorsHistory[indexVelocityVectorEnd]  )
        # 移動ベクトル ΔVelocityVector = velocityVectorEnd - velocityVectorBegin
        deltaVelocityVectorNp = velocityVectorEndNp - velocityVectorBeginNp

        # 変化してなければNoneを返す
        notChanged = (deltaVelocityVectorNp == numpy.array([0,0]))
        if notChanged.all():
            return None
        # 移動していれば、速度ベクトル = 移動ベクトル * 係数 / 母数
        else:
            accelerationVectorNp = deltaVelocityVectorNp * coAcceleration / float(population)
            accelerationVector   = tuple(accelerationVectorNp)
            return accelerationVector

def getAccelerationVector(positionHistory, population=2, numFramesDelay=0):
    pop = int(population / 2)  # 切り捨て
    if len(positionHistory) < 1+2*pop+numFramesDelay \
            or positionHistory[-1-numFramesDelay] is None \
            or positionHistory[-1-pop-numFramesDelay] is None \
            or positionHistory[-1-2*pop-numFramesDelay] is None:
        return None
    else:
        # [-1-pop]から[-1-2*pop]のときの速度
        velocity0 = getVelocityVector(positionHistory, pop, pop+numFramesDelay)
        # [-1]から[-1-pop]のときの速度
        velocity1 = getVelocityVector(positionHistory, pop, numFramesDelay)
        if velocity0 is not None and velocity1 is not None:

            printVector('v0', velocity0)
            printVector('v1', velocity1)

            v0np = numpy.array(velocity0)
            v1np = numpy.array(velocity1)
            dvnp = v1np - v0np  # v1 - v0 = Δv
            # 速度変化してなければNoneを返す
            areSameVelocity_array = (dvnp == numpy.array([0,0]))
            if areSameVelocity_array.all():
                return None
            else:
                dvnp = dvnp * 10.0 / pop
                vector = tuple(dvnp)

                printVector('a ', vector)

                return vector

def getAccelerationVectorStartStop(
        positionHistory,
        population=6,
        numFramesDelay=3,
        coForceVectorStrength=25.0):

    ### 静止判定

    # v6 - v3 = Δv3 = a3
    #
    v6 = getVelocityVector(positionHistory, 3, 0+numFramesDelay)
    v3 = getVelocityVector(positionHistory, 3, 3+numFramesDelay)

    v6np = numpy.array([0,0]) if v6 is None else numpy.array(v6)
    v3np = numpy.array([0,0]) if v3 is None else numpy.array(v3)

    v6size = math.sqrt(v6np[0]**2 + v6np[1]**2)
    v3size = math.sqrt(v3np[0]**2 + v3np[1]**2)

    if 20 < math.fabs(v6size - v3size) and (v6size < 2.0 or v3size < 2.0):
        # print '静止／急発進した ' + str(int(vSizeAfter - vSizeBefore))
        a3np = (v6np - v3np) * coForceVectorStrength / 3
        # 加速度が0ならNoneを返す
        areSameVelocity_array = (a3np == numpy.array([0,0]))
        if areSameVelocity_array.all():
            return None
        else:
            vector = tuple(a3np)
            return 'quickMotion', vector
    else:
        return 'usual'

def getAccelerationVectorFirFilter(
        positionHistory,
        population=6,
        numFramesDelay=3,
        coForceVectorStrength=25.0):
    # populationVelocityは6
    # v_6 - v_0 = Δv0 = a_0
    v11 = getVelocityVector(positionHistory, 6, numFramesDelay)
    v10 = getVelocityVector(positionHistory, 6, population+numFramesDelay)
    if v11 is None or v10 is None:
        pass
    else:
        v11np = numpy.array(v11)
        v10np = numpy.array(v10)
        anp = (v11np - v10np) * coForceVectorStrength / population
        # 加速度が0ならNoneを返す
        areSameVelocity_array = (anp == numpy.array([0,0]))
        if areSameVelocity_array.all():
            return None
        else:
            vector = tuple(anp)
            return vector

def printVector(name, tuple):
    tupleInt = (int(tuple[0]), int(tuple[1]))
    # print name + ': ' + str(tupleInt)

def getAccelerationVectorVelocitySensitive(positionHistory):
    # positionHistory[-6]とpositionHistory[-7]の
    # あいだの距離が40ピクセル以上のときは母数2で加速度を求める
    vVector = getVelocityVector(positionHistory, 1, 5)
    if vVector is None:
        pass
    elif 40 < math.sqrt(vVector[0]**2 + vVector[1]**2):
        # print '40 < v'
        return getAccelerationVector(positionHistory, 6, 3)
    else:
        return getAccelerationVector(positionHistory, 12, 0)

def cvArrow(img, pt, vector, lengthTimes, color, thickness=1, lineType=8, shift=0):
    if int(vector[0]) == 0 and int(vector[1]) == 0:
        pass
    else:
        pt1 = pt
        pt2 = (int(pt1[0] + vector[0]*lengthTimes),
               int(pt1[1] + vector[1]*lengthTimes))
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

def cvVerticalArrow(img, x, vector, lengthTimes, color, isSigned=False, thickness=1, lineType=8, shift=0):
    vx, vy = vector
    if isSigned:
        verticalVector = (0, -vx)
        baseY = img.shape[0] * 2 / 3  # 画面の下から1/3の高さ
    else:
        verticalVector = (0, -math.sqrt(vx ** 2 + vy ** 2))
        baseY = img.shape[0] - 20  # 画面下端から20px上
    cvArrow(img, (x, baseY), verticalVector,
            lengthTimes, color, thickness, lineType, shift)

def cvLine(img, pt1, pt2, color, thickness=1):
    pt1 = (int(pt1[0]), int(pt1[1]))
    pt2 = (int(pt2[0]), int(pt2[1]))
    cv2.line(img, pt1, pt2, color, thickness)

# 追跡中と検出中に呼ばれるのでメソッドにしている
def drawVelocityVectorsInStrobeMode(frameToDisplay, positionHistory,
                                    numFramesDelay, numStrobeModeSkips,
                                    velocityVectorsHistory,
                                    color=BLUE, thickness=5):
    for i in range(len(positionHistory) - numFramesDelay - 1):
        if i % numStrobeModeSkips == 0 and \
                        velocityVectorsHistory[i] is not None:
            cvArrow(
                frameToDisplay,
                positionHistory[i - numFramesDelay],
                velocityVectorsHistory[i],
                4, color, thickness
            )
            # if shouldDrawVelocityVectorsVerticallyInStrobeMode:
            #     cvVerticalArrow(
            #         frameToDisplay, spaceBetweenVerticalVectors*i,
            #         velocityVectorsHistory[i],
            #         4, color, isSigned, thickness
            #     )

def drawVelocityVectorsVerticallyInStrobeMode(frameToDisplay, positionHistory,
                                              velocityVectorsHistory, numFramesDelay,
                                              numStrobeModeSkips, spaceBetweenVerticalVectors,
                                              color=BLUE, thickness=5, isSigned=False):
    for i in range(len(positionHistory) - numFramesDelay - 1):
        if i % numStrobeModeSkips == 0 and \
                        velocityVectorsHistory[i] is not None:
            cvVerticalArrow(
                frameToDisplay, spaceBetweenVerticalVectors*i,
                velocityVectorsHistory[i],
                4, color, isSigned, thickness
            )


# 力ベクトルを描画する
def drawForceVector(img, aclVector, positionAclBegin, gravityStrength):
    if aclVector is None:
        aclVector = (0,0)
    # 加速度ベクトル - 重力ベクトル = 力ベクトル
    vector = (aclVector[0], aclVector[1] - gravityStrength)

    if vector is not None:
        cvArrow(img, positionAclBegin, vector, 1, BLUE, 5)

def getComponentVector(vector, axis):
    if vector is None:
        return None
    elif axis is "x":
        return (vector[0], 0)  # x成分のみ使う
    elif axis is "y":
        return (0, vector[1])  # y成分のみ使う
    else:
        raise ValueError('axis is neither x nor y')

class fpsWithTick(object):
    def __init__(self):
        self._count     = 0
        self._oldCount  = 0
        self._freq      = 1000 / cv2.getTickFrequency()
        self._startTime = cv2.getTickCount()
    def get(self):
        nowTime         = cv2.getTickCount()
        diffTime        = (nowTime - self._startTime) * self._freq
        self._startTime = nowTime
        fps             = (self._count - self._oldCount) / (diffTime / 1000.0)
        self._oldCount  = self._count
        self._count     += 1
        fpsRounded      = round(fps, 1)
        return fpsRounded

def pasteRect(src, dst, frameToPaste, dstRect, interpolation = cv2.INTER_LINEAR):
    """
    入力画像の部分矩形画像をリサイズして出力画像の部分矩形に貼り付ける
    :param src:     入力画像
    :type  src:     numpy.ndarray
    :param dst:     出力画像
    :type  dst:     numpy.ndarray
    :param srcRect: (x, y, w, h)
    :type  srcRect: tuple
    :param dstRect: (x, y, w, h)
    :type  dstRect: tuple
    :param interpolation: 補完方法
    :return: None
    """

    height, width, _ = frameToPaste.shape
    # x0, y0, w0, h0 = 0, 0, width, height

    x1, y1, w1, h1 = dstRect

    # コピー元の部分矩形画像をリサイズしてコピー先の部分矩形に貼り付ける
    src[y1:y1+h1, x1:x1+w1] = \
        cv2.resize(frameToPaste[0:height, 0:width], (w1, h1), interpolation = interpolation)
    # Python: cv.Resize(src, dst, interpolation=CV_INTER_LINEAR) → None
    # Parameters:
    # src – input image.
    # dst – output image; it has the size dsize (when it is non-zero) or
    # the size computed from src.size(), fx, and fy; the type of dst is the same as of src.
    # dsize –
    # output image size; if it equals zero, it is computed as:
    # dsize = Size(round(fx*src.cols), round(fy*src.rows))
    # Either dsize or both fx and fy must be non-zero.
    # fx –
    # scale factor along the horizontal axis; when it equals 0, it is computed as
    # (double)dsize.width/src.cols
    # fy –
    # scale factor along the vertical axis; when it equals 0, it is computed as
    # (double)dsize.height/src.rows
    # interpolation –
    # interpolation method:
    # INTER_NEAREST - a nearest-neighbor interpolation
    # INTER_LINEAR - a bilinear interpolation (used by default)
    # INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
    # INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
    # INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood

    dst[:] = src