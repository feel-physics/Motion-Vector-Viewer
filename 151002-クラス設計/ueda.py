from abc import *
import cv2

class ElementBase(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def MoveTo(self, x, y):
    raise NotImplementedError()
  @abstractmethod
  def Draw(self, frame):
    raise NotImplementedError()

class PointElement(ElementBase):
  def __init__(self):
    self._x = None
    self._y = None

  def MoveTo(self, x, y):
    self._x = x
    self._y = y

  def Draw(self, frame):
    cv2.circle(frame, (self._x, self._y), radius=10, color=(255,255,255))

class LineElement(ElementBase):
  def __init__(self):
    self._lastX = 0
    self._lastY = 0
    self._x     = 0
    self._y     = 0

  def MoveTo(self, x, y):
    self._lastX = self._x
    self._lastY = self._y
    self._x     = x
    self._y     = y

  def Draw(self, frame):
    cv2.line(frame, (self._lastX, self._lastY), (self._x, self._y),
             color=(255,255,255))

def DrawPolyline(frame, element, points):
  for point in points:
    element.MoveTo(point[0], point[1])
    element.Draw(frame)

if __name__ == "__main__":
  frame = cv2.imread("black.png")
  points = [(100, 100), (200, 300), (500, 300)]
  DrawPolyline(frame, PointElement(), points)
  DrawPolyline(frame, LineElement() , points)
  cv2.imwrite("frame.png", frame)
  while True:
    cv2.imshow("WinName", frame)
    cv2.waitKey(1)