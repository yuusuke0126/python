import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import time

# from math import sin, cos, tan, asin, acos, pi, sqrt, atan, e
from numpy import sin, cos, tan, arcsin, arccos, pi, sqrt, arctan, e

t = np.arange(0,10,0.1)
t = np.linspace(pi/2, 0)
t1 = 2 * arctan(tan(t/2.0)/e)

plt.plot(t*180/pi,t1*180/pi)
plt.show()