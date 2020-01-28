import csv
import numpy as np
import matplotlib.pyplot as plt

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


dist_wall = 0.76
r = 1.2
dist_stop = dist_wall + r
dist_on_curve = dist_wall + r * (1.0 - np.cos(np.pi/4.0))
dist = np.array([
  [dist_stop, dist_wall],
  [dist_on_curve, dist_on_curve],
  [dist_wall, dist_stop]
  ])

dist_goal = 0.5

with open('sample.csv') as f:
  reader = csv.reader(f)
  l = [[float(v) for v in row] for row in reader]

area = np.array(l)
area = np.append(area, np.reshape(area[0], (1,2)), axis=0)
v = np.array([(area[i+1]-area[i])/np.linalg.norm(area[i+1]-area[i]) for i in range(4)])
v90 = np.array([rotate(v[i]) for i in range(4)])

check_points = []

for i in range(4):
  for j in range(3):
    flag, check_point = calc_cross_point(area[i]+dist[j,0]*v90[i],   area[(i+1)%4]+dist[j,0]*v90[i],
                                         area[i]+dist[j,1]*v90[i-1], area[(i-1)%4]+dist[j,1]*v90[i-1])
    if flag:
      check_points = np.append(check_points, np.reshape(check_point, (1,2)))
    else:
      print('Error!')
    
check_points = np.reshape(check_points, (12,2))

goal_points = np.array([check_points[2] + dist_goal * v[0], check_points[9] - dist_goal * v[2]])

theta = [np.arctan2(v[0,1], v[0,0]), np.arctan2(v[2,1], v[2,0])]
goal_angle = np.array([
  [0, 0, np.sin(theta[0]/2.0), np.cos(theta[0]/2.0)],
  [0, 0, np.sin(theta[1]/2.0), np.cos(theta[1]/2.0)]
  ])

check_points = np.append(check_points, np.reshape(check_points[0], (1,2)), axis=0)
plt.plot(area[:,0], area[:,1], 'k')
plt.plot(check_points[:,0],check_points[:,1], '-ro')
plt.plot(goal_points[:,0], goal_points[:,1], 'bx')
plt.arrow(goal_points[0,0],goal_points[0,1], v[0,0], v[0,1], width=0.05,head_width=0.12,head_length=0.15)
plt.arrow(goal_points[1,0],goal_points[1,1], v[2,0], v[2,1], width=0.05,head_width=0.12,head_length=0.15)
plt.axis('equal')
plt.show()