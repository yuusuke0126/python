import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import time

from math import sin, cos, asin, acos, pi, sqrt, tan
# from scipy.integrate import odeint

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
p_dist = 0.15  # distance between keycart wheel center and pivot point of keyconnect
c_dist = 0.60  # distance between pivot point of keyconnect and cargo wheel center, case of 4 wheels free
# c_dist = 0.15  # distance between pivot point of keyconnect and cargo wheel center, case of front wheels free
# c_dist = 1.25  # distance between pivot point of keyconnect and cargo wheel center, case of rear wheels free

max_angle = 60.0*pi/180.0
a = pi/2.0 - max_angle

x0 = 0.0  # initial x of keycart wheel center
y0 = 0.0  # initial y of keycart wheel center

init_theta = 0.0 # initial keycart angle
# init_theta = max_angle # initial keycart angle
init_phi = 0.0  # initial cargo angle
goal_theta = pi
target_dist = 1.0
# r = c_dist*cos(a) + p_dist+tan(a)
# r = 1.35  # curve radius, calculated by geometrical method
r = 0.78  # curve radius, calculated by geometrical method
v = 0.5  # linear velocity
w = v / r  # angular velocity

dt = 0.002

curve_step = int(abs((goal_theta - init_theta) / (w*dt))) + 10
straight_step = int(abs(target_dist / (v*dt))) + 10
step_num = curve_step + straight_step
x = np.zeros(step_num)
y = np.zeros(step_num)
px = np.zeros(step_num)
py = np.zeros(step_num)
theta = np.zeros(step_num)
phi = np.zeros(step_num)

x[0], y[0], theta[0], phi[0] = x0, y0, init_theta, init_phi
px[0] = x[0] + (-p_dist) * cos(theta[0])
py[0] = y[0] + (-p_dist) * sin(theta[0])

for i in range(curve_step-1):
  xd = v*cos(theta[i])
  yd = v*sin(theta[i])
  thetad = w
  pxd = v*cos(theta[i]) - (-p_dist)*w*sin(theta[i])
  pyd = v*sin(theta[i]) + (-p_dist)*w*cos(theta[i])
  phid = (-pxd * sin(phi[i]) + pyd * cos(phi[i])) / c_dist
  x[i+1] = x[i] + xd * dt
  y[i+1] = y[i] + yd * dt
  theta[i+1] = theta[i] + thetad * dt
  px[i+1] = px[i] + pxd * dt
  py[i+1] = py[i] + pyd * dt
  phi[i+1] = phi[i] + phid * dt

w = 0.0
for i in range(curve_step-1, step_num-1):
  xd = v*cos(theta[i])
  yd = v*sin(theta[i])
  thetad = w
  pxd = v*cos(theta[i]) - p_dist*w*sin(theta[i])
  pyd = v*sin(theta[i]) + p_dist*w*cos(theta[i])
  phid = (-pxd * sin(phi[i]) + pyd * cos(phi[i])) / c_dist
  x[i+1] = x[i] + xd * dt
  y[i+1] = y[i] + yd * dt
  theta[i+1] = theta[i] + thetad * dt
  px[i+1] = px[i] + pxd * dt
  py[i+1] = py[i] + pyd * dt
  phi[i+1] = phi[i] + phid * dt

# t = range(x.shape[0])
# theta_t = odeint(func, theta0, t, args=(xd, yd))

fig = plt.figure()
ims = []

kc_rect = np.array([[    0,  0.71, 0.71,    0,     0],
                    [-0.25, -0.25, 0.25, 0.25, -0.25]])
area_rect = np.array([[0, 8, 8, 0, 0], 
                      [0, 0, 5, 5, 0]])
cargo_rect = np.array([[-1.98, -0.18, -0.18, -1.98, -1.98],
                       [-0.30, -0.30,  0.30,  0.30, -0.30]])
# cargo_rect = np.array([[-1.28, -0.18, -0.18, -1.28, -1.28],
#                        [-0.40, -0.40,  0.40,  0.40, -0.40]])
rack_rect = np.array([[2.125, 5.875, 5.875, 2.125, 2.125],
                      [  1.9,   1.9,   3.1,   3.1,   1.9]])
box1_rect = np.array([[7.55, 8.00, 8.00, 7.55, 7.55],
                      [4.55, 4.55, 5.00, 5.00, 4.55]])
box2_rect = np.array([[7.55, 8.00, 8.00, 7.55, 7.55],
                      [0.00, 0.00, 0.45, 0.45, 0.00]])
zed_pose = np.array([[0.55], 
                     [0.02]])
connect_line = np.array([[0, -p_dist], 
                         [0, 0]])
pivot_line = np.array([[0, -c_dist], 
                         [0, 0]])

cargo_rear_right = np.zeros([2,x.shape[0]])
kc_front_left = np.zeros([2,x.shape[0]])
dist = np.zeros(x.shape)
zed_dist1 = np.zeros(x.shape)
x_min = 0.0
x_max = 0.0
y_min = 0.0
y_max = 0.0

for i in range(x.shape[0]):
  kc_rect_angle = np.zeros([2,5])
  cargo_rect_angle = np.zeros([2,5])
  kc_rect_angle = rotate(kc_rect, theta[i])
  cargo_rect_angle = rotate(cargo_rect, phi[i])
  connect_angle = rotate(connect_line, theta[i])
  pivot_angle = rotate(pivot_line, phi[i])
  zed_position = np.array([[x[i]], [y[i]]]) + rotate(zed_pose, theta[i])
  zed_fov1 = zed_position + rotate(np.array([[1.0],[0.0]]), theta[i]+np.arctan(0.8/1.5))
  zed_dists = []
  for j in range(3):
    flag, fov = calc_cross_point(zed_position,zed_fov1,area_rect[:,j],area_rect[:,j+1])
    if flag:
      zed_dists.append(np.linalg.norm(zed_position-fov))
  # flag1, fov11 = calc_cross_point(zed_position,zed_fov1,area_rect[:,1],area_rect[:,2])
  # flag2, fov12 = calc_cross_point(zed_position,zed_fov1,area_rect[:,2],area_rect[:,3])
  # flag3, fov13 = calc_cross_point(zed_position,zed_fov1,area_rect[:,0],area_rect[:,1])

  zed_dist1[i] = min(zed_dists)
  cargo_rear_right[:,i] = np.array([cargo_rect_angle[0,0]+x[i], cargo_rect_angle[1,0]+y[i]])
  kc_front_left[:,i] = np.array([kc_rect_angle[0,2]+x[i], kc_rect_angle[1,2]+y[i]])
  dist[i] = sqrt((5.875-cargo_rear_right[0,i])**2 + (1.9-cargo_rear_right[1,i])**2)
  x_min_i = min(min(kc_rect_angle[0]) + x[i], min(cargo_rect_angle[0]) + px[i])
  x_max_i = max(max(kc_rect_angle[0]) + x[i], max(cargo_rect_angle[0]) + px[i])
  y_min_i = min(min(kc_rect_angle[1]) + y[i], min(cargo_rect_angle[1]) + py[i])
  y_max_i = max(max(kc_rect_angle[1]) + y[i], max(cargo_rect_angle[1]) + py[i])
  x_min = min(x_min, x_min_i)
  x_max = max(x_max, x_max_i)
  y_min = min(y_min, y_min_i)
  y_max = max(y_max, y_max_i)
  x_array = [x_min, x_max, x_max, x_min, x_min]
  y_array = [y_min, y_min, y_max, y_max, y_min]
  if i % int(round(0.1/dt)) == 0:
    kc = plt.fill(kc_rect_angle[0]+x[i], kc_rect_angle[1]+y[i], color='b')
    cargo = plt.fill(cargo_rect_angle[0]+px[i], cargo_rect_angle[1]+py[i], color='r')
    connect = plt.plot(connect_angle[0]+x[i], connect_angle[1]+y[i], linewidth=4.0, color='k')
    pivot = plt.plot(pivot_angle[0]+px[i], pivot_angle[1]+py[i], linewidth=4.0, color='k')
    area = plt.plot(x_array, y_array, color='m')
    # for k in reversed(range(10)):
    #   l = (i - i%int(round(0.1/dt)*5)) + k*int(round(0.1/dt))*5
    #   if l > x.shape[0] - 1:
    #     l = x.shape[0] - 1
    #   r_v = sqrt(xd[l]**2+yd[l]**2)
    #   if r_v < 0.1:
    #     r_v = 0.1
    #   if k==0:
    #     l = i
    #     r_v = sqrt(xd[l]**2+yd[l]**2)
    #     if r_v < 0.1:
    #       r_v = 0.1
    #     arr = plt.arrow(x[l], y[l], r_v*cos(theta[l]), r_v*sin(theta[l]), width=0.01,head_width=0.08,head_length=0.12, color='r')
    #   else:
    #     arr = plt.arrow(x[l], y[l], r_v*cos(theta[l]), r_v*sin(theta[l]), width=0.005,head_width=0.08,head_length=0.12, color='k')
    #   arrs.append(arr)
    ims.append(kc+cargo+connect+pivot+area)

# print("minimum distance between cargo to inside rack: {:.2f} [m]".format(min(dist)))
# print("minimum distance between keycart to wall: {:.2f} [m]".format(min(8.0-max(kc_front_left[0,:]),min(kc_front_left[1,:]))))
# print("minimum distance between keycart to box: {:.2f} [m]".format(min(np.array([np.linalg.norm(kc_front_left[:,i]-np.array([7.55, 0.45])) for i in range(x.shape[0])]))))
# print("minimum distance between cargo to wall: {:.2f} [m]".format(kc2wall-0.33))
# print("stop point: {:.2f} [m]".format(stop_dist - 0.71))
print("curve radius: %1.2f [m]" % (r))
print("x_dist: %1.2f [m], y_dist: %1.2f [m]" % (x_max - x_min, y_max - y_min))
print("max angle: %1.2f [deg]" % (max(abs(theta-phi)*180/pi)))
ani = animation.ArtistAnimation(fig, ims, interval=0.1*1000/2, repeat=False)
plt.axis('equal')
plt.show(block=True)
# ani.save("output.gif", writer="imagemagick")


