# Python + OpenCVで
# 画像処理

---
## いちばん簡単なOpenCVを使ったプログラム

```
import cv2

image = cv2.imread('hoge.png')
cv2.imwrite('hoge.jpg', image)
```

---
## 

```
success, frame = videoCapture.read()
while success: # フレームがなくなるまで繰り返す
    videoWriter.write(frame)
    success, frame = videoCapture.read()
```