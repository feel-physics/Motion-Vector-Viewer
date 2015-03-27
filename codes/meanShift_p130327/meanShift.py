# coding=utf-8
__author__ = 'weed'

import cv2
import numpy as np


def main():
    # カメラのキャプチャー
    cap = cv2.VideoCapture(0)
    # 最初のフレームを取得
    ret,frame = cap.read()
    # 追跡したい領域の初期設定
    r,h,c,w = 50,200,300,300
    track_window = (c,r,w,h)
    # 追跡のためのROIを設定
    roi = frame[r:r+h, c:c+w]
    # HSV色空間に変換
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # マスク画像の作成
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    # ヒストグラムの計算
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    # ヒストグラムの正規化
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    # 終了基準の設定
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    while(1):
        # フレームの取得
        ret ,frame = cap.read()
        # HSV色空間に変換
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # バックプロジェクションの計算
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # 新しい場所を取得するためにmeanshiftを適用
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # 追跡している領域を描く
        x,y,w,h = track_window
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,0,200),2)
        # 結果の表示
        cv2.imshow("Mean Shift Method",frame)

        # 任意のキーが押されたら終了
        if cv2.waitKey(10) > 0:
            # キャプチャー解放
            cap.release()
            # ウィンドウ破棄
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()