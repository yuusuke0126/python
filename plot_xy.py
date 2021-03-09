import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import time

from math import sin, cos, asin, acos, pi, sqrt

t = np.arange(0,10,0.1)
x0 = 100.0
y0 = 20.0

x = x0 * np.exp(-1*t)
y = y0 * np.exp(-1*t)

plt.plot(x,y)
plt.show()