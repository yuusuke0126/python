from enum import IntEnum
import numpy as np

from math import sin, cos, asin, acos, pi, sqrt, tan
# from scipy.integrate import odeint

class Mode(IntEnum):
    STANDBY = 0
    APPROACH = 1
    RETRY_APPROACH = 2
    GRIP = 3
    RETRY_GRIP = 4
    FINISH = 5
    FAILED = 6

def calc_cross_point(pointA, pointB, pointC, pointD):
  cross_point = (0,0)
  bunbo = (pointB[0] - pointA[0]) * (pointD[1] - pointC[1]) - (pointB[1] - pointA[1]) * (pointD[0] - pointC[0])
  # Case in lines are parallel
  if (bunbo == 0):
      return False, cross_point
  vectorAC = ((pointC[0] - pointA[0]), (pointC[1] - pointA[1]))
  r = ((pointD[1] - pointC[1]) * vectorAC[0] - (pointD[0] - pointC[0]) * vectorAC[1]) / bunbo
  distance = ((pointB[0] - pointA[0]) * r, (pointB[1] - pointA[1]) * r)
  cross_point = np.array([pointA[0] + distance[0], pointA[1] + distance[1]])
  return True, cross_point

def rotate(v, theta=-np.pi/2.0):
  rotateMat = np.array([
      [np.cos(theta), -np.sin(theta)],
      [np.sin(theta), np.cos(theta)],
  ])
  return np.dot(rotateMat, v)

def calc_norm2(v):
  return v[0]*v[0] + v[1]*v[1]

def segment_distance_point(z, cp1, cp2):
  v = cp1 - cp2
  px, py = z
  x1, y1 = cp1
  x2, y2 = cp2
  a = x2 - x1
  b = y2 - y1
  a2 = a * a
  b2 = b * b
  r2 = a2 + b2
  tt = -(a * (x1 - px) + b * (y1 - py))
  if calc_norm2(v) < 0.000001:
    return (x1 - px) * (x1 - px) + (y1 - py) * (y1 - py), cp1
  if tt < 0:
    return (x1 - px) * (x1 - px) + (y1 - py) * (y1 - py), cp1
  if tt > r2:
    return (x2 - px) * (x2 - px) + (y2 - py) * (y2 - py), cp2

  f1 = a * (y1 - py) - b * (x1 - px)
  return f1 * f1 / r2, cp1 + np.array([a, b]) * tt / r2

def segment_distance_segment(a1, a2, b1, b2):
  a = a2 - a1
  b = b2 - b1
  ab1 = b1 - a1
  ab2 = b2 - a1
  ba1 = a1 - b1
  ba2 = a2 - b1
  if calc_norm2(a) < 0.000001:
    # case vector a is point
    dist, point = segment_distance_point(a1, b1, b2)
    return dist, a1, point
  elif calc_norm2(b) < 0.000001:
    # case vector b is point
    dist, point = segment_distance_point(b1, a1, a2)
    return dist, point, b1
  elif calc_displacement(a, ab1) * calc_displacement(a, ab2) > 0 or calc_displacement(b, ba1) * calc_displacement(b, ba2) > 0:
    # case vectors have no intersection
    p_min_a = a1
    l_min, p_min_b = segment_distance_point(a1, b1, b2)
    l, p = segment_distance_point(a2, b1, b2)
    if l < l_min:
      l_min = l
      p_min_a = a2
      p_min_b = p
    l, p = segment_distance_point(b1, a1, a2)
    if l < l_min:
      l_min = l
      p_min_a = p
      p_min_b = b1
    l, p = segment_distance_point(b2, a1, a2)
    if l < l_min:
      l_min = l
      p_min_a = p
      p_min_b = b2
    return l_min, p_min_a, p_min_b
  else:
    denom = calc_displacement(a, b)
    ua = (b[0]*ba1[1] - b[1]*ba1[0]) / denom
    p = a1 + ua * a
    return 0.0, p, p

def calc_displacement(v1, v2):
  return v1[0]*v2[1] - v1[1]*v2[0]

def new_func():
    print("Test function called!")

class TestModule():
    def __init__(self) -> None:
        print('TestModule class initialized!')
        self.mode = Mode.STANDBY

    def test_func(self):
        print(self.mode)
        new_func()
    
    def test_super_func(self):
        print("Super function!")

class TestModuleChild(TestModule):
    def test_func(self):
        print("Child function!")
        super().test_super_func()
        return super().test_func()

if __name__ == '__main__':
  try:
    tm = TestModuleChild()
    tm.test_func()
  except Exception as ex:
    print(ex)