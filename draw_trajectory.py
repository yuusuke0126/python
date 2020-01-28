import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import time

from math import sin, cos, asin, acos, pi, sqrt
from scipy.integrate import odeint

plt.close('all')
c_dist = 1.22 # distance between keycart wheel center and cargo wheel center

# angle = 45*pi/180
angle = 0
# r = c_dist / sin(angle)
r = 1.2
kc2wall = 0.76
stop_dist = kc2wall + r * (1 - sin(angle))
max_x = 0.5
max_z = 0.5
dt = 0.002

x = np.arange(1.7, 8.0 - stop_dist, dt*max_x)
y = (5.0 - kc2wall) * np.ones(x.shape)
z = np.zeros(x.shape)

x = np.append(x, 8.0 - stop_dist)
y = np.append(y, y[-1])
z = np.append(z, z[-1])


z_ap = np.arange(z[-1], -angle, -dt*max_z)
z_ap = np.append(z_ap, -angle)
x_ap = x[-1] * np.ones(z_ap.shape)
y_ap = y[-1] * np.ones(z_ap.shape)

x = np.append(x, x_ap)
y = np.append(y, y_ap)
z = np.append(z, z_ap)


z_ap = np.arange(z[-1], -pi/2, -dt*min(max_z, max_x/r))
z_ap = np.append(z_ap, -pi/2)

x_ap = x[-1] + r * (-np.sin(z_ap)+sin(z[-1]))
y_ap = y[-1] + r * (np.cos(z_ap)-cos(z[-1]))
x_ap[-1] = (8.0 - kc2wall)
y_ap[-1] = y[-1] - r * cos(angle)

x = np.append(x, x_ap)
y = np.append(y, y_ap)
z = np.append(z, z_ap)


y_ap = np.arange(y[-1], stop_dist, -dt*max_x)
y_ap = np.append(y_ap, stop_dist)
x_ap = x[-1] * np.ones(y_ap.shape)
z_ap = z[-1] * np.ones(y_ap.shape)

x = np.append(x, x_ap)
y = np.append(y, y_ap)
z = np.append(z, z_ap)


z_ap = np.arange(z[-1], -pi/2-angle, -dt*max_z)
z_ap = np.append(z_ap, -pi/2-angle)
x_ap = x[-1] * np.ones(z_ap.shape)
y_ap = y[-1] * np.ones(z_ap.shape)

x = np.append(x, x_ap)
y = np.append(y, y_ap)
z = np.append(z, z_ap)


z_ap = np.arange(z[-1], -pi, -dt*min(max_z, max_x/r))
z_ap = np.append(z_ap, -pi)

x_ap = x[-1] + r * (-np.sin(z_ap)+sin(z[-1]))
y_ap = y[-1] + r * (np.cos(z_ap)-cos(z[-1]))
x_ap[-1] = x[-1] - r * cos(angle)
y_ap[-1] = kc2wall

x = np.append(x, x_ap)
y = np.append(y, y_ap)
z = np.append(z, z_ap)

x_ap = np.arange(x[-1], stop_dist, -dt*max_x)
x_ap = np.append(x_ap, stop_dist)
y_ap = [y[-1]] * np.ones(x_ap.shape)
z_ap = [z[-1]] * np.ones(x_ap.shape)

x = np.append(x, x_ap)
y = np.append(y, y_ap)
z = np.append(z, z_ap)

x = np.append(x, x[-1]*np.ones((int(round(0.1/dt))*2,)))
y = np.append(y, y[-1]*np.ones((int(round(0.1/dt))*2,)))
z = np.append(z, z[-1]*np.ones((int(round(0.1/dt))*2,)))

xd = (x[1:] - x[0:-1]) / dt
yd = (y[1:] - y[0:-1]) / dt
zd = (z[1:] - z[0:-1]) / dt
xd = np.append(xd, 0)
yd = np.append(yd, 0)
zd = np.append(zd, 0)

# theta0 = 0.0

theta = []
theta = np.append(theta, 0)

for t in range(x.shape[0]-1):
  thetad = 1.0 / c_dist * (-xd[t] * sin(theta[-1]) + yd[t] * cos(theta[-1]))
  theta = np.append(theta, theta[-1]+thetad*dt)


# t = range(x.shape[0])
# theta_t = odeint(func, theta0, t, args=(xd, yd))

fig = plt.figure()
ims = []

kc_rect = np.array([[    0,  0.71, 0.71,    0,     0],
                    [-0.25, -0.25, 0.25, 0.25, -0.25]])
area_rect = np.array([[0, 8, 8, 0, 0], 
                      [0, 0, 5, 5, 0]])
cargo_rect = np.array([[-1.29, -0.45, -0.45, -1.29, -1.29],
                       [-0.33, -0.33,  0.33,  0.33, -0.33]])
rack_rect = np.array([[2.125, 5.875, 5.875, 2.125, 2.125],
                      [  1.9,   1.9,   3.1,   3.1,   1.9]])
box1_rect = np.array([[7.55, 8.00, 8.00, 7.55, 7.55],
                      [4.55, 4.55, 5.00, 5.00, 4.55]])
box2_rect = np.array([[7.55, 8.00, 8.00, 7.55, 7.55],
                      [0.00, 0.00, 0.45, 0.45, 0.00]])

cargo_rear_right = np.zeros([2,x.shape[0]])
kc_front_left = np.zeros([2,x.shape[0]])
dist = np.zeros(x.shape)

for i in range(x.shape[0]):
  kc_rect_angle = np.zeros([2,5])
  cargo_rect_angle = np.zeros([2,5])
  R1 = np.array([[cos(z[i]), -sin(z[i])],
                [sin(z[i]),  cos(z[i])]])
  R2 = np.array([[cos(theta[i]), -sin(theta[i])],
                 [sin(theta[i]),  cos(theta[i])]])
  for j in range(5):
    kc_rect_angle[:,j] = np.dot(R1, kc_rect[:,j])
    cargo_rect_angle[:,j] = np.dot(R2, cargo_rect[:,j])
  if i % int(round(0.1/dt)) == 0:
    area = plt.plot(area_rect[0], area_rect[1], color='k')
    rack = plt.plot(rack_rect[0], rack_rect[1], color='m')
    box1 = plt.plot(box1_rect[0], box1_rect[1], color='k')
    box2 = plt.plot(box2_rect[0], box2_rect[1], color='k')
    kc = plt.fill(kc_rect_angle[0]+x[i], kc_rect_angle[1]+y[i], color='b')
    cargo = plt.fill(cargo_rect_angle[0]+x[i], cargo_rect_angle[1]+y[i], color='r')
    connect = plt.plot([x[i], x[i]+np.average(cargo_rect_angle[0,1:3])],[y[i], y[i]+np.average(cargo_rect_angle[1,1:3])], linewidth=3.0, color='k')
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
    ims.append(area+kc+cargo+connect+arrs+box1+box2)
  cargo_rear_right[:,i] = np.array([cargo_rect_angle[0,0]+x[i], cargo_rect_angle[1,0]+y[i]])
  kc_front_left[:,i] = np.array([kc_rect_angle[0,2]+x[i], kc_rect_angle[1,2]+y[i]])
  dist[i] = sqrt((5.875-cargo_rear_right[0,i])**2 + (1.9-cargo_rear_right[1,i])**2)

print("minimum distance between cargo to inside rack: {:.2f} [m]".format(min(dist)))
print("minimum distance between keycart to wall: {:.2f} [m]".format(min(8.0-max(kc_front_left[0,:]),min(kc_front_left[1,:]))))
print("minimum distance between keycart to box: {:.2f} [m]".format(min(np.array([np.linalg.norm(kc_front_left[:,i]-np.array([7.55, 0.45])) for i in range(x.shape[0])]))))
print("minimum distance between cargo to wall: {:.2f} [m]".format(kc2wall-0.33))
print("stop point: {:.2f} [m]".format(stop_dist - 0.71))
print("max angle: {:.1f} [deg]".format(max(abs(theta-z)*180/pi)))
ani = animation.ArtistAnimation(fig, ims, interval=0.1*1000/2, repeat=False)
plt.axis('equal')
plt.show(block=False)
# ani.save("output.gif", writer="imagemagick")

