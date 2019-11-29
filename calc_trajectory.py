
from math import sin, cos, asin, acos, pi
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
  
  
def func(theta, phi, D, relateve_angle0):
  # relateve_angle0 = theta0 - (phi0 + pi/2)
  p = 1.7 # Distance between keycart wheel center and pallete cart wheel center
  r = D / (1 + cos(relateve_angle0))
  dtheta_dphi = r / p * cos(theta - phi)
  return dtheta_dphi
  
theta0 = pi/2
phi0 = 0.0
phi_end = pi
D = 5.1

relative_angle0 = theta0 - (phi0 + pi/2)

phi = np.linspace(phi0, phi_end, 50)
  
theta_t = odeint(func, theta0, phi, args=(D,relative_angle0))
print("Calculation finished!")

keycart_pose = phi+pi/2
pallet_pose = theta_t[:,0]
print('Last relative angle: {:.1f}'.format((keycart_pose[-1]-pallet_pose[-1])*180/pi))

plt.plot(phi, (keycart_pose-pallet_pose)*180/pi)
plt.show()