import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import time

from math import sin, cos, asin, acos, pi, sqrt

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

plt.close('all')

## trajectory curve radius
r = 1.3

## cargo size
cargo_length = 1.28
cargo_width = 0.42
key_connect_length = 0.45
cargo_connect_length = 0.09
c_dist = cargo_length / 2 + key_connect_length # distance between keycart wheel center and cargo wheel center
c_dist2 = cargo_length / 2 + cargo_connect_length # distance between 2nd cargo connect point and 2nd cargo wheel center

## simulation settings
max_x = 0.5 # max lin_x velocity
max_z = 0.5 # max ang_z velocity
dt = 0.002 # simulation sampling time

## initial position of keycart x, y, and z=angle
x = np.array([1.3])
y = np.array([0.0])
z = np.array([pi/2])

## keycart motion setting, just curve and straight
z_ap = np.arange(z[-1], pi, dt*min(max_z, max_x/r))
z_ap = np.append(z_ap, pi)

x_ap = x[-1] + r * (np.sin(z_ap)-np.sin(z[-1]))
y_ap = y[-1] + r * (-np.cos(z_ap)+np.cos(z[-1]))

x = np.append(x, x_ap)
y = np.append(y, y_ap)
z = np.append(z, z_ap)

x_ap = np.arange(x[-1], -7, -dt*max_x)
x_ap = np.append(x_ap, -7)
y_ap = [y[-1]] * np.ones(x_ap.shape)
z_ap = [z[-1]] * np.ones(x_ap.shape)

x = np.append(x, x_ap)
y = np.append(y, y_ap)
z = np.append(z, z_ap)

x = np.append(x, x[-1]*np.ones((int(round(0.1/dt))*2,)))
y = np.append(y, y[-1]*np.ones((int(round(0.1/dt))*2,)))
z = np.append(z, z[-1]*np.ones((int(round(0.1/dt))*2,)))


## calculate velocity from position
xd = (x[1:] - x[0:-1]) / dt
yd = (y[1:] - y[0:-1]) / dt
zd = (z[1:] - z[0:-1]) / dt
xd = np.append(xd, 0)
yd = np.append(yd, 0)
zd = np.append(zd, 0)

## calculate 1st cargo angle
theta = []
theta = np.append(theta, pi/2)
for t in range(x.shape[0]-1):
  thetad = 1.0 / c_dist * (-xd[t] * sin(theta[-1]) + yd[t] * cos(theta[-1]))
  theta = np.append(theta, theta[-1]+thetad*dt)

## calculate 2nd cargo connect point coodinate
x2 = []
y2 = []
c2_v = np.array([[-key_connect_length-cargo_length-cargo_connect_length], [0.0]])
for t in range(x.shape[0]):
  v = rotate(c2_v, theta[t])
  x2 = np.append(x2, x[t] + v[0])
  y2 = np.append(y2, y[t] + v[1])

## calculate 2nd cargo connect point velocity
x2d = (x2[1:] - x2[0:-1]) / dt
y2d = (y2[1:] - y2[0:-1]) / dt
z2d = (theta[1:] - theta[0:-1]) / dt
x2d = np.append(x2d, 0)
y2d = np.append(y2d, 0)
z2d = np.append(z2d, 0)

## calculate 2nd cargo angle
theta2 = []
theta2 = np.append(theta2, pi/2)
for t in range(x.shape[0]-1):
  theta2d = 1.0 / c_dist2 * (-x2d[t] * sin(theta2[-1]) + y2d[t] * cos(theta2[-1]))
  theta2 = np.append(theta2, theta2[-1]+theta2d*dt)

## plot settings
fig = plt.figure(figsize=(8.0, 8.0))
ims = []

kc_rect = np.array([[    0,  0.71, 0.71,    0,     0],
                    [-0.25, -0.25, 0.25, 0.25, -0.25]])
area_rect = np.array([[-10, 2.2, 2.2, -10, -10], 
                      [-10, -10, 2.2, 2.2, -10]])
rack_rect = np.array([[-10, 0, 0, -10, -10],
                      [-10, -10, 0, 0, -10]])
cargo_rect = np.array([[-cargo_length-key_connect_length, -key_connect_length, -key_connect_length, -cargo_length-key_connect_length, -cargo_length-key_connect_length],
                       [-cargo_width/2, -cargo_width/2,  cargo_width/2,  cargo_width/2, -cargo_width/2]])
cargo2_rect = np.array([[-cargo_length-cargo_connect_length, -cargo_connect_length, -cargo_connect_length, -cargo_length-cargo_connect_length, -cargo_length-cargo_connect_length], 
                        [-cargo_width/2, -cargo_width/2,  cargo_width/2,  cargo_width/2, -cargo_width/2]])

cargo_rear_right = np.zeros([2,x.shape[0]])
kc_front_left = np.zeros([2,x.shape[0]])
dist = np.zeros(x.shape)
zed_dist1 = np.zeros(x.shape)

for i in range(x.shape[0]):
  kc_rect_angle = np.zeros([2,5])
  cargo_rect_angle = np.zeros([2,5])
  kc_rect_angle = rotate(kc_rect, z[i])
  cargo_rect_angle = rotate(cargo_rect, theta[i])
  cargo2_rect_angle = rotate(cargo2_rect, theta2[i])

  if i % int(round(0.1/dt)) == 0:
    area = plt.plot(area_rect[0], area_rect[1], color='k')
    rack = plt.plot(rack_rect[0], rack_rect[1], color='m')
    kc = plt.fill(kc_rect_angle[0]+x[i], kc_rect_angle[1]+y[i], color='b')
    cargo = plt.fill(cargo_rect_angle[0]+x[i], cargo_rect_angle[1]+y[i], color='r')
    cargo2 = plt.fill(cargo2_rect_angle[0]+x2[i], cargo2_rect_angle[1]+y2[i], color='g')
    connect = plt.plot([x[i], x[i]+np.average(cargo_rect_angle[0,1:3])],[y[i], y[i]+np.average(cargo_rect_angle[1,1:3])], linewidth=3.0, color='k')
    c2 = plt.plot(x2[i], y2[i], marker="*", color='k')
    arrs = []
    for k in reversed(range(10)):
      l = (i - i%int(round(0.1/dt)*5)) + k*int(round(0.1/dt))*5
      if l > x.shape[0] - 1:
        l = x.shape[0] - 1
      r_v = sqrt(xd[l]**2+yd[l]**2)
      if r_v < 0.1:
        r_v = 0.1
      if k==0:
        l = i
        r_v = sqrt(xd[l]**2+yd[l]**2)
        if r_v < 0.1:
          r_v = 0.1
        arr = plt.arrow(x[l], y[l], r_v*cos(z[l]), r_v*sin(z[l]), width=0.01,head_width=0.08,head_length=0.12, color='r')
      else:
        arr = plt.arrow(x[l], y[l], r_v*cos(z[l]), r_v*sin(z[l]), width=0.005,head_width=0.08,head_length=0.12, color='k')
      arrs.append(arr)
    ims.append(area+kc+cargo+connect+arrs+c2+cargo2)
  cargo_rear_right[:,i] = np.array([cargo_rect_angle[0,0]+x[i], cargo_rect_angle[1,0]+y[i]])
  kc_front_left[:,i] = np.array([kc_rect_angle[0,2]+x[i], kc_rect_angle[1,2]+y[i]])
  dist[i] = sqrt((5.875-cargo_rear_right[0,i])**2 + (1.9-cargo_rear_right[1,i])**2)

# print("minimum distance between cargo to inside rack: {:.2f} [m]".format(min(dist)))
# print("minimum distance between keycart to wall: {:.2f} [m]".format(min(8.0-max(kc_front_left[0,:]),min(kc_front_left[1,:]))))
# print("minimum distance between keycart to box: {:.2f} [m]".format(min(np.array([np.linalg.norm(kc_front_left[:,i]-np.array([7.55, 0.45])) for i in range(x.shape[0])]))))
# print("minimum distance between cargo to wall: {:.2f} [m]".format(kc2wall-cargo_width/2))
# print("stop point: {:.2f} [m]".format(stop_dist - 0.71))
# print("max angle: {:.1f} [deg]".format(max(abs(theta-z)*180/pi)))
ani = animation.ArtistAnimation(fig, ims, interval=0.1*1000/2, repeat=False)
plt.axis('equal')
plt.show(block=False)
# ani.save("output.gif", writer="imagemagick")

