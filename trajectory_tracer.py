import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import time

def rotate(v, theta=-np.pi/2.0):
  rotateMat = np.array([
      [np.cos(theta), -np.sin(theta)],
      [np.sin(theta), np.cos(theta)],
  ])
  return np.dot(rotateMat, v)

def check_status(z, checkpoints, status):
  tolerance = 0.1
  i = status
  target_dir = checkpoints[:,i] - checkpoints[:,i-1]
  target_dir = target_dir / np.linalg.norm(target_dir)
  vec = z - checkpoints[:,i]
  if np.dot(vec, target_dir) > - tolerance:
    i += 1
  new_status = i
  return new_status

def calc_input(z, theta, checkpoint):
  # d = 0.1  # distance between center point of drive wheels and control point
  v_max = 0.5
  w_max = 1.5
  B_inv = np.array([[d*np.cos(theta), d*np.sin(theta)], 
                    [-np.sin(theta),  np.cos(theta)]])
  e = z - checkpoint
  u = -1 * np.dot(B_inv, e)
  if u[0] != 0:
    u[1] = u[1] / abs(u[0]) * v_max
    u[0] = u[0] / abs(u[0]) * v_max
  if abs(u[1]) > w_max:
    u[0] = u[0] / abs(u[1]) * w_max
    u[1] = u[1] / abs(u[1]) * w_max
  return u

def input_smoother(u, prev_u, duration):
  acc_lim_v = 0.3
  acc_lim_w = 1.0
  v = u[0]; w = u[1]
  prev_v = prev_u[0]; prev_w = prev_u[1]
  if duration > 0.5:
    prev_v = 0.0; prev_w = 0.0
    duration = 0.1
  if abs(v-prev_v)/duration > acc_lim_v:
    v = prev_v + np.sign(v-prev_v) * acc_lim_v * duration
  if abs(w-prev_w)/duration > acc_lim_w:
    w = prev_w + np.sign(w-prev_w) * acc_lim_w * duration
  return np.array([v, w])

def update_status(z, theta, u, dt):
  # d = 0.1  # distance between center point of drive wheels and control point
  v = u[0]; w = u[1]
  if w != 0:
    z_w = z - d * np.array([np.cos(theta), np.sin(theta)])
    curve_vec = v/w * np.array([np.cos(theta+np.pi/2.0), np.sin(theta+np.pi/2.0)])
    new_z_w = z_w + curve_vec - rotate(curve_vec, w*dt)
    new_z = new_z_w + d * np.array([np.cos(theta+w*dt), np.sin(theta+w*dt)])
  else:
    new_z = z + v * np.array([np.cos(theta), np.sin(theta)]) * dt
  return new_z, theta+w*dt

plt.close('all')

d = 0.5

z = np.zeros(2)
theta = 0.0
u = np.zeros(2)

right_shift = -1.5
length = 6.5
# checkpoints = np.array([[0.0, abs(right_shift), length-abs(right_shift), length, 7, 5], 
#                         [0.0,     -right_shift,            -right_shift,    0.0, 3, 5]])
checkpoints = np.array([[0.0, 2, 5, 6.5, 6.8, 7.0,  7, 0], 
                        [0.0, 0, 0, 0.2, 0.3, 0.7, 4, 4]])

dt = 0.05
max_T = 100.0

T = np.arange(0.0, max_T, dt)

status = 1

Z = np.zeros((2,len(T)+1))
Theta = np.zeros(len(T)+1)
U = np.zeros((2,len(T)+1))

Z[:,0] = z; Theta[0] = theta; U[:,0] = u

i=0
for t in T:
  u = calc_input(z, theta, checkpoints[:,status])
  u = input_smoother(u, U[:,i], dt)
  z, theta = update_status(z, theta, u, dt)
  U[:,i+1] = u; Z[:,i+1] = z; Theta[i+1] = theta
  status = check_status(z, checkpoints, status)
  if status == checkpoints.shape[1]:
    break
  i+=1

if i < len(T)+1:
  Z = np.delete(Z,np.s_[i:], 1); U = np.delete(U, np.s_[i:], 1); Theta = np.delete(Theta, np.s_[i:])

fig = plt.figure()
ims = []

kc_rect = np.array([[   -d, 0.71-d, 0.71-d,   -d,    -d],
                    [-0.25,  -0.25,   0.25, 0.25, -0.25]])
wheel_left = np.array([[-0.1-d, 0.1-d, 0.1-d, -0.1-d, -0.1-d],
                       [  0.25,  0.25,   0.3,    0.3,   0.25]])
wheel_right = np.array([[-0.1-d, 0.1-d, 0.1-d, -0.1-d, -0.1-d],
                        [ -0.25, -0.25,  -0.3,   -0.3,  -0.25]])
area_rect = np.array([[-1.0,    8, 8, -1.0, -1.0], 
                      [-1.0, -1.0, 5,    5, -1.0]])
# cargo_rect = np.array([[-1.29, -0.45, -0.45, -1.29, -1.29],
#                        [-0.33, -0.33,  0.33,  0.33, -0.33]])
# rack_rect = np.array([[2.125, 5.875, 5.875, 2.125, 2.125],
#                       [  1.9,   1.9,   3.1,   3.1,   1.9]])
# box1_rect = np.array([[7.55, 8.00, 8.00, 7.55, 7.55],
#                       [4.55, 4.55, 5.00, 5.00, 4.55]])
# box2_rect = np.array([[7.55, 8.00, 8.00, 7.55, 7.55],
#                       [0.00, 0.00, 0.45, 0.45, 0.00]])

for i in range(Z.shape[1]):
  kc_rect_angle = np.zeros([2,5])
  wheel_left_angle = np.zeros([2,5])
  wheel_right_angle = np.zeros([2,5])
  # cargo_rect_angle = np.zeros([2,5])
  kc_rect_angle = rotate(kc_rect, Theta[i])
  wheel_left_angle = rotate(wheel_left, Theta[i])
  wheel_right_angle = rotate(wheel_right, Theta[i])
  # cargo_rect_angle = rotate(cargo_rect, theta[i])
  if i % int(round(0.1/dt)) == 0:
    area = plt.plot(area_rect[0], area_rect[1], color='k')
    kc = plt.fill(kc_rect_angle[0]+Z[0,i], kc_rect_angle[1]+Z[1,i], color='b', alpha=0.5)
    wl = plt.fill(wheel_left_angle[0]+Z[0,i], wheel_left_angle[1]+Z[1,i], color='grey', alpha=0.5)
    wr = plt.fill(wheel_right_angle[0]+Z[0,i], wheel_right_angle[1]+Z[1,i], color='grey', alpha=0.5)
    cpm = [plt.scatter(checkpoints[0], checkpoints[1], color='r')]
    traj = plt.plot(Z[0,:i], Z[1,:i], color='k')
    zp = [plt.scatter(Z[0,i], Z[1,i], color='k')]
    ims.append(area+kc+wl+wr+cpm+traj+zp)

ani = animation.ArtistAnimation(fig, ims, interval=0.1*1000/2, repeat=False)
plt.axis('equal')
plt.show(block=False)
fig2 = plt.figure()
plt.plot(T[:i-1], U[0,:i-1], label='lin_v')
plt.plot(T[:i-1], U[1,:i-1], label='ang_z')
plt.legend()
plt.show(block=False)
# ani.save("output.gif", writer="imagemagick")

