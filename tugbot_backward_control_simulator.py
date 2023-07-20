import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import time, yaml
import tqdm
from test_module import segment_distance_segment, rotate

from math import sin, cos, asin, acos, pi, sqrt, tan

def update_status(t, Z, P, Theta, Phi, v=0.5, w=0.0):
  v += np.random.randn()*0.01
  w += np.random.randn()*0.01
  xd = v*cos(Theta[t])
  yd = v*sin(Theta[t])
  thetad = w
  pxd = v*cos(Theta[t]) - (-p_dist)*w*sin(Theta[t])
  pyd = v*sin(Theta[t]) + (-p_dist)*w*cos(Theta[t])
  phid = (-pxd * sin(Phi[t]) + pyd * cos(Phi[t])) / c_dist
  Z[0, t+1] = Z[0, t] + xd * dt
  Z[1, t+1] = Z[1, t] + yd * dt
  Theta[t+1] = Theta[t] + thetad * dt
  P[0, t+1] = P[0, t] + pxd * dt
  P[1, t+1] = P[1, t] + pyd * dt
  Phi[t+1] = Phi[t] + phid * dt

margin = 0.5
cf_dist = 0.45
cart_length = 0.8
d = 0.3
D = 0.6
c = D / d + 1.0

zcd_x = margin+cf_dist+cart_length - d - D - 3.0
zcd = np.array([zcd_x, 0.0])

Kp = np.array([[1.0, 0.0],
                [0.0, 4.0]])
Kt = 2.0

def calc_B12(phi):
  sin2_phi = sin(phi)**2.0
  sin_2phi = sin(2.0*phi)
  B12 = np.array([[1.0 - c*sin2_phi,       c/2.0*sin_2phi],
                  [  c/2.0*sin_2phi, 1.0 - c + c*sin2_phi]])
  return B12

def calc_input(z, p, theta, phi):
  v = -0.3
  # phi += np.random.randn()*0.01
  angle =  theta - phi
  if abs(sin(angle)) > 0.0001:
    r = D / sin(angle)
    w = v / r
  else:
    w = 0.0
  zc = z - (d+D) * np.array([cos(phi), sin(phi)])
  e = zc - zcd
  v_r = - np.dot(calc_B12(phi), np.dot(Kp, e))
  vx = v_r[0]; vy = v_r[1]
  theta_d = np.arctan2(vy, vx) + pi
  et = (theta - theta_d) % (2.0*pi)
  if et > pi:
    et -= 2*pi
  if abs(et) > 15*pi/180:
    w = -et
    v = 0.0
  else:
    w += -et
  return v, w, theta_d

def check_finish(z, phi):
  zc = z - (d+D) * np.array([cos(phi), sin(phi)])
  e = zc - zcd
  if abs(e[0]) < 0.1:
    return True, 0.0
  else:
    return False, abs(e[0]) - 0.1

config_file_name = "EV_stab"
area_file_name = "EV_back"
print(config_file_name + "\n" + area_file_name)
SAVE_FLAG = True
plt.close('all')
with open('config/' + config_file_name + '.yaml', 'r') as yml:
  config = yaml.safe_load(yml)

p_dist = config['p_dist']  # distance between tugbot wheel center and pivot point of gripper
c_dist = config['c_dist']  # distance between pivot point of gripper and cargo wheel center

cf_dist = config['cf_dist']  # distance between pivot point of gripper and cart front edge
cart_length = config['cart_length']  # for cargo cart
cart_width = config['cart_width']  # for cargo cart

cr_dist = -(cf_dist + cart_length)  # distance between pivot point of gripper and cart rear edge
cw_2 = cart_width / 2.0

max_angle = 60.0*pi/180.0
a = pi/2.0 - max_angle

motion_plan = config['motion_plan']

x0 = config['x0']  # initial x of tugbot wheel center
y0 = config['y0']  # initial y of tugbot wheel center

init_theta = config['init_theta'] * pi / 180 # initial tugbot angle
init_phi = config['init_phi'] * pi / 180  # initial cargo angle

dt = 0.002
T_max = 60.0
t_num = int(T_max / dt)

Z = np.zeros((2,t_num))
P = np.zeros((2,t_num))
Theta = np.zeros(t_num)
Phi = np.zeros(t_num)
Zc = np.zeros((2,t_num))
U = np.zeros((2,t_num))
Theta_d = np.zeros(t_num)

Z[0,0], Z[1,0], Theta[0], Phi[0] = x0, y0, init_theta, init_phi
P[0,0] = Z[0,0] + (-p_dist) * cos(Theta[0])
P[1,0] = Z[1,0] + (-p_dist) * sin(Theta[0])

print("Start simulation...")

sampling_t = 0.05
finish_flag, ex0 = check_finish(Z[:,0], Phi[0])
for t in range(t_num-1):
  Zc[:,t] = Z[:,t] - (d+D) * np.array([cos(Phi[t]), sin(Phi[t])])
  finish_flag, ex = check_finish(Z[:,t], Phi[t])
  if t % 500 == 0:
    progress = int((ex0-ex)/ex0*100)
    print("\r", "progress: %3d %%, rest:  %1.2f [m] / %1.2f [m]" % (progress, ex, ex0), end="")
  if finish_flag:
    Z = np.delete(Z, np.s_[t+1:], 1)
    Zc = np.delete(Zc, np.s_[t+1:], 1)
    U = np.delete(U, np.s_[t+1:], 1)
    Theta = np.delete(Theta, np.s_[t+1:])
    Theta_d = np.delete(Theta_d, np.s_[t+1:])
    Phi = np.delete(Phi, np.s_[t+1:])
    progress = int((ex0-ex)/ex0*100)
    print("\r", "progress: %3d %%, rest:  %1.2f [m] / %1.2f [m]" % (progress, ex, ex0))
    break
  if t % int(sampling_t / dt) == 0:
    v, w, theta_d = calc_input(Z[:,t], P[:,t], Theta[t], Phi[t])
  U[:,t] = np.array([v, w])
  Theta_d[t] = theta_d
  update_status(t, Z, P, Theta, Phi, v, w)

print("\nSimulation finished!")

print("Start analysis...")

E = np.zeros(Zc.shape)
E = Zc - zcd.reshape(2,-1)
Theta = Theta % (2*pi)
Theta[Theta > pi] = Theta[Theta > pi] - 2*pi
Theta_d = Theta_d % (2*pi)
Theta_d[Theta_d > pi] = Theta_d[Theta_d > pi] - 2*pi
Et = (Theta - Theta_d) % (2*pi)
Et[Et > pi] = Et[Et > pi] - 2*pi
Ud = np.zeros(U.shape)
Ud[:,:-2] = (U[:,1:-1] - U[:,:-2]) / dt

fig, ax = plt.subplots()
ims = []
with open('config/areas/' + area_file_name + '.csv') as file:
  area_num = int(sum(1 for line in file) / 2)
area_rect_list = []
for i in range(area_num):
  area_rect = np.loadtxt('config/areas/' + area_file_name + '.csv', delimiter=',', skiprows=2*i, max_rows=2)
  area_rect_list.append(area_rect)
kc_rect = np.array([[-0.27,  0.27, 0.27, -0.27, -0.27],
                    [-0.30, -0.30, 0.30,  0.30, -0.30]])  # tugbot
# kc_rect = np.array([[-0.09,  0.61, 0.61, -0.09, -0.09],
#                     [-0.25, -0.25, 0.25,  0.25, -0.25]])  # keycart
cargo_rect = np.array([[cr_dist, -cf_dist, -cf_dist, cr_dist, cr_dist],
                       [  -cw_2,    -cw_2,     cw_2,    cw_2,   -cw_2]])  # general purpose
connect_line = np.array([[0, -p_dist], 
                         [0, 0]])
pivot_line = np.array([[0, -c_dist], 
                         [0, 0]])

t_num_sim = Z.shape[1]
for i in tqdm.tqdm(range(t_num_sim)):
  if i % int(round(0.1/dt)) == 0:
    kc_rect_angle = np.zeros([2,5])
    cargo_rect_angle = np.zeros([2,5])
    kc_rect_angle = rotate(kc_rect, Theta[i])
    cargo_rect_angle = rotate(cargo_rect, Phi[i])
    connect_angle = rotate(connect_line, Theta[i])
    pivot_angle = rotate(pivot_line, Phi[i])
    kc = ax.fill(kc_rect_angle[0]+Z[0,i], kc_rect_angle[1]+Z[1,i], color='b')
    cargo = ax.fill(cargo_rect_angle[0]+P[0,i], cargo_rect_angle[1]+P[1,i], color='r')
    connect = ax.plot(connect_angle[0]+Z[0,i], connect_angle[1]+Z[1,i], linewidth=4.0, color='k')
    pivot = ax.plot(pivot_angle[0]+P[0,i], pivot_angle[1]+P[1,i], linewidth=4.0, color='k')
    # min_dist_line = ax.plot([min_point_robot[0], min_point_area[0]], [min_point_robot[1], min_point_area[1]], color='r')
    areas = None
    for j in range(area_num):
      area = ax.plot(area_rect_list[j][0], area_rect_list[j][1], color='m')
      areas = area if areas is None else areas + area
    im = kc+cargo+connect+pivot+areas
    ims.append(im)

stop_t = t_num_sim * dt
T = np.linspace(0.0, stop_t, t_num_sim)

print("Analysis finished!")
print("max angle: %1.2f [deg]" % (max(abs(Theta-Phi)*180/pi)))
print("Making animation...")
ani = animation.ArtistAnimation(fig, ims, interval=0.1*1000/2, repeat=False)
plt.axis('equal')
plt.show(block=True)
fig, axes = plt.subplots(2,2, tight_layout=True)
axes[0,0].plot(T, E[0,:], label="Ex [m]")
axes[0,0].plot(T, E[1,:], label="Ey [m]")
axes[0,0].set_xlabel("t [sec]")
axes[0,0].legend()
axes[0,1].plot(T, U[1,:], label="Uw [rad/s]")
axes[0,1].plot(T, Ud[1,:], label="Udw [rad/s^2]")
axes[0,1].set_xlabel("t [sec]")
axes[0,1].legend()
axes[1,0].plot(T, Theta*180/pi, label="theta [deg]")
axes[1,0].plot(T, Theta_d*180/pi, label="theta_d [deg]")
axes[1,0].plot(T, Et*180/pi, label="e_t [deg]")
axes[1,0].set_xlabel("t [sec]")
axes[1,0].legend()
axes[1,1].plot(T, (Theta-Phi)*180/pi, label="relative angle [deg]")
axes[1,1].set_xlabel("t [sec]")
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