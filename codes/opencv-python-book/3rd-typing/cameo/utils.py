# coding=utf-8
__author__ = 'weed'

import cv2
import numpy
import math

def getVelocityVector(passedPoints, population=1, numDelayFrames=0):
    # populationは母集団。すなわち、何フレーム分の位置データを用いて速度を求めるか。
    # indexRewindは加速度を求めるのに使う
    if len(passedPoints) < population+numDelayFrames+1 \
            or passedPoints[-1-numDelayFrames] is None \
            or passedPoints[-1-population-numDelayFrames] is None:
        return None
    else:
        # 最後からpopulation個前の点 pt0
        pt0np = numpy.array(passedPoints[-(1+population+numDelayFrames)])
        # 最後の点 pt1
        pt1np = numpy.array(passedPoints[-(1+numDelayFrames)])
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

def getAccelerationVector(passedPoints, population=2, numDelayFrames=0):
    pop = int(population / 2)  # 切り捨て
    if len(passedPoints) < 1+2*pop+numDelayFrames \
            or passedPoints[-1-numDelayFrames] is None \
            or passedPoints[-1-pop-numDelayFrames] is None \
            or passedPoints[-1-2*pop-numDelayFrames] is None:
        return None
    else:
        # [-1-pop]から[-1-2*pop]のときの速度
        velocity0 = getVelocityVector(passedPoints, pop, pop+numDelayFrames)
        # [-1]から[-1-pop]のときの速度
        velocity1 = getVelocityVector(passedPoints, pop, numDelayFrames)
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

def getAccelerationVectorFirFilter(passedPoints, population=2, numDelayFrames=5):
    # at frame 12: a6 = v7 - v6 = ((Pt7 - Pt2) - (Pt6 - Pt1))
    # at frame 12: a7 = v8 - v6 = ((Pt8 - Pt3) - (Pt6 - Pt1)) / 2
    # at frame 12: a8 = v9 - v6 = ((Pt9 - Pt4) - (Pt6 - Pt1)) / 3
    v1 = getVelocityVector(passedPoints, 6, 6-population)
    v0 = getVelocityVector(passedPoints, 6, 6)
    if v1 is None or v0 is None:
        pass
    # # TODO: 次の3行はHard Coded
    # if len(passedPoints) < 2*population+numDelayFrames \
    #         or passedPoints[1-population*2-numDelayFrames] is None \
    #         or passedPoints[2-population*2-numDelayFrames] is None \
    #         or passedPoints[  population  -numDelayFrames] is None \
    #         or passedPoints[1-population  -numDelayFrames] is None:
    #     return None
    else:
        v1np = numpy.array(v1)
        v0np = numpy.array(v0)
        # pt1np = numpy.array(passedPoints[1-population*2-numDelayFrames])  # 12-11=1
        # pt2np = numpy.array(passedPoints[2-population*2-numDelayFrames])  # 12-10=2
        # pt6np = numpy.array(passedPoints[ -population  -numDelayFrames])  # 12-6 =6
        # pt7np = numpy.array(passedPoints[1-population  -numDelayFrames])  # 12-5 =7
        # a6np  = ((pt7np - pt2np) - (pt6np - pt1np)) * 10.0 / population
        anp = (v1np - v0np) * 50.0 / population
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