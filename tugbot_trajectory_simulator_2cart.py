import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import time, yaml
import tqdm
from test_module import segment_distance_segment, rotate

from math import sin, cos, asin, acos, pi, sqrt, tan

def update_status(t, z, p, p2, theta, phi, phi2, v=0.5, w=0.0):
  xd = v*cos(theta[t])
  yd = v*sin(theta[t])
  thetad = w
  pxd = xd - (-p_dist)*w*sin(theta[t])
  pyd = yd + (-p_dist)*w*cos(theta[t])
  if c_dist == 0.0:
    phid = thetad
  else:
    phid = (-pxd * sin(phi[t]) + pyd * cos(phi[t])) / c_dist
  p2xd = pxd - (-p2_dist)*phid*sin(phi[t])
  p2yd = pyd + (-p2_dist)*phid*cos(phi[t])
  if c2_dist == 0.0:
    phi2d = phid
  else:
    phi2d = (-p2xd * sin(phi2[t]) + p2yd * cos(phi2[t])) / c2_dist
  z[0, t+1] = z[0, t] + xd * dt
  z[1, t+1] = z[1, t] + yd * dt
  theta[t+1] = theta[t] + thetad * dt
  p[0, t+1] = p[0, t] + pxd * dt
  p[1, t+1] = p[1, t] + pyd * dt
  phi[t+1] = phi[t] + phid * dt
  p2[0, t+1] = p2[0, t] + p2xd * dt
  p2[1, t+1] = p2[1, t] + p2yd * dt
  phi2[t+1] = phi2[t] + phi2d * dt

config_file_name = "2x2_2cargo_stab_goal"
area_file_name = "2x2_tel"
print(config_file_name + "\n" + area_file_name)
SAVE_FLAG = True
DIST_FLAG = False
plt.close('all')
with open('config/' + config_file_name + '.yaml', 'r') as yml:
  config = yaml.safe_load(yml)

p_dist = config['p_dist']  # distance between tugbot wheel center and pivot point of gripper
c_dist = config['c_dist']  # distance between pivot point of gripper and cargo wheel center
p2_dist = config['p2_dist']  # distance between pivot point of gripper and pivot point of 2nd cart
c2_dist = config['c2_dist']  # distance between pivot point of 2nd cart and 2nd cart wheel center

cf_dist = config['cf_dist']  # distance between pivot point of gripper and cart front edge
cf2_dist = config['cf2_dist']  # distance between pivot point of 2nd cart and 2nd cart front edge
cart_length = config['cart_length']  # for cargo cart
cart_width = config['cart_width']  # for cargo cart

cr_dist = -(cf_dist + cart_length)  # distance between pivot point of gripper and cart rear edge
cr2_dist = -(cf2_dist + cart_length)  # distance between pivot point of 2nd cart and 2nd cart rear edge
cw_2 = cart_width / 2.0

max_angle = 60.0*pi/180.0
a = pi/2.0 - max_angle

motion_plan = config['motion_plan']

x0 = config['x0']  # initial x of tugbot wheel center
y0 = config['y0']  # initial y of tugbot wheel center

init_theta = config['init_theta'] * pi / 180 # initial tugbot angle
init_phi = config['init_phi'] * pi / 180  # initial cargo angle
init_phi2 = config['init_phi2'] * pi / 180  # initial 2nd cart angle

z = np.zeros((2,1))
p = np.zeros((2,1))
p2 = np.zeros((2,1))
theta = np.zeros(1)
phi = np.zeros(1)
phi2 = np.zeros(1)

z[0,0], z[1,0], theta[0], phi[0], phi2[0] = x0, y0, init_theta, init_phi, init_phi2
p[0,0] = z[0,0] + (-p_dist) * cos(theta[0])
p[1,0] = z[1,0] + (-p_dist) * sin(theta[0])
p2[0,0] = p[0,0] + (-p2_dist) * cos(phi[0])
p2[1,0] = p[1,0] + (-p2_dist) * sin(phi[0])

dt = 0.002
print("Start simulation...")
for i in range(len(motion_plan)):
  motion_type = motion_plan[i]["type"]
  goal = motion_plan[i]["goal"]
  if motion_type == "linear":
    v = np.sign(goal) * 0.5
    w = 0.0
    step_num = np.ceil(abs(goal / (v*dt))).astype(int)
  elif motion_type == "curve":
    goal = goal * pi / 180
    r = motion_plan[i]["curve_radius"]
    v = np.sign(r) * 0.5
    w = np.sign(goal) * abs(v / r)
    step_num = np.ceil(abs(goal / (w*dt))).astype(int)
  elif motion_type == "spin":
    goal = goal * pi / 180
    v = 0.0
    w = np.sign(goal) * 1.0
    step_num = np.ceil(abs(goal / (w*dt))).astype(int)
  current_t = z.shape[1]
  z = np.hstack((z, np.zeros((2, step_num))))
  p = np.hstack((p, np.zeros((2, step_num))))
  p2 = np.hstack((p2, np.zeros((2, step_num))))
  theta = np.hstack((theta, np.zeros(step_num)))
  phi = np.hstack((phi, np.zeros(step_num)))
  phi2 = np.hstack((phi2, np.zeros(step_num)))
  for t in range(current_t-1, current_t+step_num-1):
    update_status(t, z, p, p2, theta, phi, phi2, v, w)
  if motion_type == "linear":
    dist = np.linalg.norm(z[:,current_t] - z[:,-2])
    v = (abs(goal) - dist) / dt * np.sign(goal)
    w = 0.0
  elif motion_plan == "curve" or motion_plan == "spin":
    dist_theta = theta[-2] - theta[current_t]
    rest = goal - dist_theta
    rest = rest % (2*pi)
    if rest > pi:
      rest -= 2*pi
    w = rest / dt
    if motion_plan == "curve":
      v = (r * w) * np.sign(goal)
    else:
      v = 0.0
  t = current_t + step_num - 1
  update_status(t-1, z, p, p2, theta, phi, phi2, v, w)

print("Simulation finished!")

print("Start analysis...")
fig, ax = plt.subplots()
ims = []
with open('config/areas/' + area_file_name + '.csv') as file:
  area_num = int(sum(1 for line in file) / 2)
area_rect_list = []
for i in range(area_num):
  area_rect = np.loadtxt('config/areas/' + area_file_name + '.csv', delimiter=',', skiprows=2*i, max_rows=2)
  area_rect_list.append(area_rect)
# print(area_rect)
kc_rect = np.array([[-0.27,  0.24, 0.24, -0.27, -0.27],
                    [-0.30, -0.30, 0.30,  0.30, -0.30]])  # tugbot
# kc_rect = np.array([[-0.09,  0.61, 0.61, -0.09, -0.09],
#                     [-0.25, -0.25, 0.25,  0.25, -0.25]])  # keycart
# cargo_rect = np.array([[-1.90, -0.90, -0.90, -1.90, -1.90, -2.125, -2.125, -2.413, -2.413, -2.125, -2.125,  -1.90, -1.90],
#                        [-0.20, -0.20,  0.20,  0.20, 0.125,  0.125,  0.060,  0.060, -0.060, -0.060, -0.125, -0.125, -0.20]])  # case for nishsin seifun cart
cargo_rect = np.array([[cr_dist, -cf_dist, -cf_dist, cr_dist, cr_dist],
                       [  -cw_2,    -cw_2,     cw_2,    cw_2,   -cw_2]])  # general purpose
cargo2_rect = np.array([[cr2_dist, -cf2_dist, -cf2_dist, cr2_dist, cr2_dist],
                       [    -cw_2,     -cw_2,      cw_2,     cw_2,    -cw_2]])  # general purpose
zed_pose = np.array([[0.55], 
                     [0.02]])
connect_line = np.array([[0, -p_dist], 
                         [0, 0]])
if c_dist == 0.0:
  pivot_line = np.array([[0, -cf_dist], 
                         [0, 0]])
else:
  pivot_line = np.array([[0, -c_dist], 
                         [0, 0]])
if c2_dist == 0.0:
  pivot2_line = np.array([[0, -cf2_dist], 
                         [0, 0]])
else:
  pivot2_line = np.array([[0, -c2_dist], 
                         [0, 0]])

cargo_rear_right = np.zeros(z.shape)
cargo_center_right = np.zeros(z.shape)
kc_front_left = np.zeros(z.shape)
dist = np.zeros(z.shape[1])
zed_dist1 = np.zeros(z.shape[1])
x_min = 0.0
x_max = 0.0
y_min = 0.0
y_max = 0.0
dist_min = -1
t_min = None

for i in tqdm.tqdm(range(z.shape[1])):
  kc_rect_angle = np.zeros([2,5])
  cargo_rect_angle = np.zeros([2,5])
  kc_rect_angle = rotate(kc_rect, theta[i])
  cargo_rect_angle = rotate(cargo_rect, phi[i])
  cargo2_rect_angle = rotate(cargo2_rect, phi2[i])
  connect_angle = rotate(connect_line, theta[i])
  pivot_angle = rotate(pivot_line, phi[i])
  pivot2_angle = rotate(pivot2_line, phi2[i])
  zed_position = z[:,i:i+1] + rotate(zed_pose, theta[i])
  zed_fov1 = zed_position + rotate(np.array([[1.0],[0.0]]), theta[i]+np.arctan(0.8/1.5))
  min_dist = None
  min_point_robot = None
  min_point_area = None
  kc_z = kc_rect_angle + z[:,i:i+1]
  cg_z = cargo_rect_angle + z[:,i:i+1]
  cg2_z = cargo2_rect_angle + p2[:,i:i+1]
  if DIST_FLAG is True:
    for j in range(area_num):
      area_rect = area_rect_list[j]
      edge_num = area_rect.shape[1] - 1
      for k in range(edge_num):
        for l in range(4):
          distance, pr, pa = segment_distance_segment(kc_z[:,l], kc_z[:,l+1], area_rect[:,k], area_rect[:,k+1])
          if min_dist is None or distance < min_dist:
            min_dist = distance
            min_point_robot = pr
            min_point_area = pa
        for l in range(4):
          distance, pr, pa = segment_distance_segment(cg_z[:,l], cg_z[:,l+1], area_rect[:,k], area_rect[:,k+1])
          if min_dist is None or distance < min_dist:
            min_dist = distance
            min_point_robot = pr
            min_point_area = pa
        for l in range(4):
          distance, pr, pa = segment_distance_segment(cg2_z[:,l], cg2_z[:,l+1], area_rect[:,k], area_rect[:,k+1])
          if min_dist is None or distance < min_dist:
            min_dist = distance
            min_point_robot = pr
            min_point_area = pa
    dist[i] = min_dist
    if dist_min > min_dist or t_min is None:
      dist_min = min_dist
      t_min = i
  x_min_i = min(min(kc_rect_angle[0]) + z[0,i], min(cargo_rect_angle[0]) + p[0,i])
  x_max_i = max(max(kc_rect_angle[0]) + z[0,i], max(cargo_rect_angle[0]) + p[0,i])
  y_min_i = min(min(kc_rect_angle[1]) + z[1,i], min(cargo_rect_angle[1]) + p[1,i])
  y_max_i = max(max(kc_rect_angle[1]) + z[1,i], max(cargo_rect_angle[1]) + p[1,i])
  x_min = min(x_min, x_min_i)
  x_max = max(x_max, x_max_i)
  y_min = min(y_min, y_min_i)
  y_max = max(y_max, y_max_i)
  x_array = [x_min, x_max, x_max, x_min, x_min]
  y_array = [y_min, y_min, y_max, y_max, y_min]
  if i % int(round(0.1/dt)) == 0:
    kc = ax.fill(kc_rect_angle[0]+z[0,i], kc_rect_angle[1]+z[1,i], color='b')
    cargo = ax.fill(cargo_rect_angle[0]+p[0,i], cargo_rect_angle[1]+p[1,i], color='r')
    cargo2 = ax.fill(cargo2_rect_angle[0]+p2[0,i], cargo2_rect_angle[1]+p2[1,i], color='r')
    connect = ax.plot(connect_angle[0]+z[0,i], connect_angle[1]+z[1,i], linewidth=4.0, color='k')
    pivot = ax.plot(pivot_angle[0]+p[0,i], pivot_angle[1]+p[1,i], linewidth=4.0, color='k')
    pivot2 = ax.plot(pivot2_angle[0]+p2[0,i], pivot2_angle[1]+p2[1,i], linewidth=4.0, color='k')
    if DIST_FLAG is True:
      min_dist_line = ax.plot([min_point_robot[0], min_point_area[0]], [min_point_robot[1], min_point_area[1]], color='r')
    else:
      min_dist_line = []
    areas = None
    for j in range(area_num):
      area = ax.plot(area_rect_list[j][0], area_rect_list[j][1], color='m')
      areas = area if areas is None else areas + area
    im = kc+cargo+cargo2+connect+pivot+pivot2+areas+min_dist_line
    ims.append(im)

if DIST_FLAG is True:
  t_min *= dt
  stop_t = len(dist) * dt
  T = np.linspace(0.0, stop_t, len(dist))
  dist = np.sqrt(dist)
  dist_min = np.sqrt(dist_min)

print("Analysis finished!")

print("minimum distance between robot/cargo to wall: {:.2f} [m]".format(dist_min))
print("x_dist: %1.2f [m], y_dist: %1.2f [m]" % (x_max - x_min, y_max - y_min))
print("max angle: %1.2f [deg]" % (max(abs(theta-phi)*180/pi)))
print("max angle: %1.2f [deg]" % (max(abs(phi2-phi)*180/pi)))
print("Making animation...")
ani = animation.ArtistAnimation(fig, ims, interval=0.1*1000/2, repeat=False)
plt.axis('equal')
plt.show(block=True)
if DIST_FLAG is True:
  plt.plot(T, dist, label="minimum distance")
  plt.plot([0,stop_t],[dist_min, dist_min], linestyle="--", label="%2.3f [m]" % (dist_min))
  plt.plot([t_min, t_min], [0,max(dist)], linestyle="--", label="%2.1f [sec]" % (t_min))
  plt.xlabel("t [sec]")
  plt.ylabel("minimum distance [m]")
  plt.legend()
plt.show()

if SAVE_FLAG:
  val = input("Do you want to save the gif? Y/n: ")
  if val == "Y" or val == "y":
    val = input("File name: ")
    if val == "":
      val = "output"
    file_name = val + ".gif"
    print("filename: " + file_name)
    print("Saving gif...")
    ani.save("gifs/" + file_name, writer="imagemagick")