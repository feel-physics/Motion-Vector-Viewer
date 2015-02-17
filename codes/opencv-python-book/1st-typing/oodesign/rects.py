# coding=utf-8
__author__ = 'weed'

import cv2

def outlineRect(image, rect, color):
    """
    (x, y, w, h)形式の矩形を元に、OpenCV形式の矩形を返す
    :param image: object
    :param rect: tuple
    :param color: object
    :return: object
    """
    if rect is None:
        return
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x+w, y+h), color)

def copyRect(src, dst, srcRect, dstRect,
             interpolation = cv2.INTER_LINEAR):
    """
    コピー元の一部をコピー先の一部にコピーする
    :param src: object
    :param dst: object
    :param srcRect: tuple
    :param dstRect: tuple
    :param interpolation: object
    :return: object
    """
    x0, y0, w0, h0 = srcRect
    x1, y1, w1, h1 = dstRect

    # コピー元の該当する矩形の内容をリサイズし、
    # コピー先の該当する矩形に結果を貼り付ける
    dst[y1:y1+h1, x1:x1+w1] = \
        cv2.resize(src[y0:y0+h0, x0:x0+w0], (w1, h1),
                   interpolation = interpolation)

def swapRects(src, dst, rects,
              interpolation = cv2.INTER_LINEAR):
    """
    2つ以上の矩形を取り替えてコピー元をコピーする
    :param src: object
    :param dst: object
    :param rects: list of [tuple]
    :param interpolation: object
    :return: object
    """
    if dst is not src:
        dst[:] = src

    numRects = len(rects)
    if numRects < 2:
        return

    # 最後の矩形の中身を、一時的な領域にコピーする
    x, y, w, h = rects[numRects - 1]
    temp = src[y:y+h, x:x+w].copy()
    """:type : object"""

    # 矩形の中身を一つずつ次の矩形にコピーする（？）
    i = numRects - 2
    while i >= 0:
        copyRect(src, dst, rects[i], rects[i+1], interpolation)
        i -= 1

    # 一時的に記録された中身を、最初の矩形にコピーする
    copyRect(temp, dst, (0, 0, w, h), rects[0], interpolation)