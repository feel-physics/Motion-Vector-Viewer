# coding=utf-8
__author__ = 'weed'

import cv2
import numpy
import math

def getVelocityVector(passedPoints, population=1, numFramesDelay=0):
    # populationは母集団。すなわち、何フレーム分の位置データを用いて速度を求めるか。
    # populationが3の場合はvはPt[-1],Pt[-4]を参照する。
    if len(passedPoints) < population+numFramesDelay+1 \
            or passedPoints[-1-numFramesDelay] is None \
            or passedPoints[-1-population-numFramesDelay] is None:
        return None
    else:
        # 最後からpopulation個前の点 pt0
        pt0np = numpy.array(passedPoints[-(1+population+numFramesDelay)])
        # 最後の点 pt1
        pt1np = numpy.array(passedPoints[-(1+numFramesDelay)])
        # 移動ベクトル Δpt = pt1 - pt0
        dptnp = pt1np - pt0np
        # 移動してなければNoneを返す
        areSamePoint_array = (dptnp == numpy.array([0,0]))
        if areSamePoint_array.all():
            return None
        else:
            dptnp = dptnp / float(population)
            vector = tuple(dptnp)
            return vector

def getAccelerationVector(passedPoints, population=2, numFramesDelay=0):
    pop = int(population / 2)  # 切り捨て
    if len(passedPoints) < 1+2*pop+numFramesDelay \
            or passedPoints[-1-numFramesDelay] is None \
            or passedPoints[-1-pop-numFramesDelay] is None \
            or passedPoints[-1-2*pop-numFramesDelay] is None:
        return None
    else:
        # [-1-pop]から[-1-2*pop]のときの速度
        velocity0 = getVelocityVector(passedPoints, pop, pop+numFramesDelay)
        # [-1]から[-1-pop]のときの速度
        velocity1 = getVelocityVector(passedPoints, pop, numFramesDelay)
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

def getAccelerationVectorStartStop(passedPoints, population=6, numFramesDelay=3):

    ### 静止判定

    # v6 - v3 = Δv3 = a3
    #
    v6 = getVelocityVector(passedPoints, 3, 0+numFramesDelay)
    v3 = getVelocityVector(passedPoints, 3, 3+numFramesDelay)

    v6np = numpy.array([0,0]) if v6 is None else numpy.array(v6)
    v3np = numpy.array([0,0]) if v3 is None else numpy.array(v3)

    v6size = math.sqrt(v6np[0]**2 + v6np[1]**2)
    v3size = math.sqrt(v3np[0]**2 + v3np[1]**2)

    if 20 < math.fabs(v6size - v3size):
        # print '静止／急発進した ' + str(int(vSizeAfter - vSizeBefore))
        a3np = (v6np - v3np) * 50.0 / 3
        # 加速度が0ならNoneを返す
        areSameVelocity_array = (a3np == numpy.array([0,0]))
        if areSameVelocity_array.all():
            return None
        else:
            vector = tuple(a3np)
            return 'startStop', vector
    else:
        return 'usual'

def getAccelerationVectorFirFilter(passedPoints, population=6, numFramesDelay=3):
    # populationVelocityは6
    # v_6 - v_0 = Δv0 = a_0
    v11 = getVelocityVector(passedPoints, 6, numFramesDelay)
    v10 = getVelocityVector(passedPoints, 6, population+numFramesDelay)
    if v11 is None or v10 is None:
        pass
    else:
        v11np = numpy.array(v11)
        v10np = numpy.array(v10)
        anp = (v11np - v10np) * 50.0 / population
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

def getAccelerationVectorVelocitySensitive(passedPoints):
    # passedPoints[-6]とpassedPoints[-7]の
    # あいだの距離が40ピクセル以上のときは母数2で加速度を求める
    vVector = getVelocityVector(passedPoints, 1, 5)
    if vVector is None:
        pass
    elif 40 < math.sqrt(vVector[0]**2 + vVector[1]**2):
        # print '40 < v'
        return getAccelerationVector(passedPoints, 6, 3)
    else:
        return getAccelerationVector(passedPoints, 12, 0)

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

def cvLine(img, pt1, pt2, color, thickness=1):
    pt1 = (int(pt1[0]), int(pt1[1]))
    pt2 = (int(pt2[0]), int(pt2[1]))
    cv2.line(img, pt1, pt2, color, thickness)

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