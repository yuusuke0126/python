
from math import sin, cos, asin, acos, pi
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from tf.transformations import euler_from_quaternion
  
# t = np.linspace(-pi,pi, 500)
# cos_t = np.cos(t)
# Y=np.zeros(t.shape)

# for i in range(len(t)):
#   if abs(1-cos_t[i]) <= 1.0:
#     Y[i] = 1.5/3.0**(1-cos_t[i]**3.0)
#   else:
#     Y[i] = -0.2 * (1-cos_t[i])**2.0 + 0.4 * (1-cos_t[i]) + 0.3

# dist = np.linspace(0, 5.0, 500)
# Y = np.zeros(dist.shape)

# for i in range(len(dist)):
#   if dist[i] < (1.5**2.0 / (0.6*0.7) / 2.0):
#     Y[i] = np.sqrt(2.0*0.6*dist[i]*0.7)
#   else:
#     Y[i] = 1.5

# c = linspace(-1.2,1.2,500)
# Y = np.zeros(c.shape)
# for i in range(len(c)):
#   Y[i] = max(0.5, -1*c[i]**2.0+1.5)

v_sm = 0.7
acc_sm = 0.4
v_max = 1.5
c = 0.5 * v_sm**2.0 / acc_sm
dist_max = (v_max + c)**2.0 *0.5 / acc_sm
t = linspace(0, 10, 5000)
dt = 10.0/5000.0
x0 = -dist_max
X = np.zeros(t.shape)
V = np.zeros(t.shape)
X[0] = x0
for i in range(len(t)-1):
  if -X[i] <= v_sm**2.0 / acc_sm:
    V[i] = -acc_sm / v_sm * X[i]
  else:
    V[i] = np.sqrt(-2.0*acc_sm*(X[i]+c))
  X[i+1] = X[i] + V[i] * dt
  # if X[i+1]+c > 0:
  #   X[i+1] = -c
V[-1] = V[-2]
A = (V[1:] - V[:-1])/dt

t = linspace(0, 10, 5000)
dt = 10.0/5000.0
theta0 = 0.0
dtheta0 = 1.0
Theta = np.zeros(t.shape)
dTheta = np.zeros(t.shape)

Theta[0] = theta0
dTheta[0] = dtheta0
for i in range(len(t)-1):
  dTheta[i+1] = 1.8*cos(Theta[i])
  if abs(dTheta[i+1]) > 1.0:
    dTheta[i+1] = dTheta[i+1] * 1.0 / abs(dTheta[i+1])
  acc = (dTheta[i+1] - dTheta[i]) / dt
  if abs(acc) > 1.5:
    acc = acc * 1.5 / abs(acc)
    dTheta[i+1] = dTheta[i] + acc * dt
  Theta[i+1] = Theta[i] + dTheta[i]*dt

ddTheta = (dTheta[1:] - dTheta[:-1])/dt

plt.plot(t, Theta)
plt.show()
plt.plot(t, dTheta)
plt.show()
plt.plot(t[:-1], ddTheta)
plt.show()