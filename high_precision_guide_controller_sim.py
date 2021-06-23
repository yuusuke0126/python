import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# d = 0.15  # distance between wheel center and control point

class WaypointTracer():
    def __init__(self) -> None:
        Kx = 1.0
        Ky = 4.0
        self.K = np.array([[Kx, 0],
                           [0, Ky]])
        

class SMC_Cotnroller():
    def __init__(self) -> None:
        self.d = 0.15  # distance between wheel center and control point
        self.a = 1.0
        self.b = 5.0
        self.alpha = 5.0
        self.S = np.array([[self.a, 0, self.d/2.0],
                    [0, self.b, self.d/2.0]])
        self.K = np.eye(2) * 0.50

    X=np.array([1.0,1.0,0.0])

    def calc_u(self, X: np.ndarray) -> np.ndarray:
        t = X[2]
        Xcp = X + np.array([self.d*np.cos(t), self.d*np.sin(t), 0])
        B = self.calc_B(t)
        u = - np.dot(np.dot(np.linalg.inv(np.dot(self.S, B)), self.K), self.sat_SX(Xcp))
        return u

    def sat_SX(self, X):
        SX = np.dot(self.S, X) * self.alpha
        if abs(SX[0]) > 1.0:
            SX[0] = SX[0] / abs(SX[0])
        if abs(SX[1]) > 1.0:
            SX[1] = SX[1] / abs(SX[1])
        return SX

    def calc_B(self, t):
        B = np.array([[np.cos(t), -self.d*np.sin(t)],
                    [np.sin(t),  self.d*np.cos(t)],
                    [0, 1]])
        return B


def calc_B(t):
    B = np.array([[np.cos(t), 0],
                  [np.sin(t), 0],
                  [        0, 1]])
    return B

def update_X(X, u, dt=0.1):
    t = X[2]
    X_dot = np.dot(calc_B(t), u)
    X2 = X + X_dot * dt
    return X2

def rotate(v, theta=-np.pi/2.0):
  rotateMat = np.array([
      [np.cos(theta), -np.sin(theta)],
      [np.sin(theta), np.cos(theta)],
  ])
  return np.dot(rotateMat, v)

if __name__ == '__main__':
    controller = SMC_Cotnroller()
    dt = 0.1
    T = 150.0
    t = np.arange(0.0, T, dt)
    t_num = len(t)
    X0 = np.array([10.0, 10.0, np.pi*2])
    X = np.zeros((3, t_num))
    X[:,0] = X0
    U = np.zeros((2, t_num))
    for i in range(t_num-1):
        U[:,i] = controller.calc_u(X[:,i])
        X[:,i+1] = update_X(X[:,i], U[:,i], dt)

    plt.close('all')
    tb_rect = np.array([[-0.2, 0.2, 0.2, -0.2, -0.2],
                        [-0.3, -0.3, 0.3, 0.3, -0.3]])
    area_rect = np.array([[min(X[0,:])-0.3, max(X[0,:])+0.3, max(X[0,:])+0.3],
                     [min(X[1,:])-0.3, min(X[0,:])-0.3, max(X[1,:])+0.3]])
    fig = plt.figure()
    ims = []
    t_num = X.shape[1]
    for i in range(t_num):
        if i % int(round(0.3/dt)) == 0:
            tb_rect_a = rotate(tb_rect, X[2,i])
            area = plt.plot(area_rect[0,:], area_rect[1,:], color='w')
            tb = plt.fill(tb_rect_a[0,:]+X[0,i], tb_rect_a[1,:]+X[1,i], color='b')
            traj = plt.plot(X[0,:i], X[1,:i], color='k')
            goal = plt.plot(0.0, 0.0, marker='*', color='k')
            ims.append(area+tb+traj+goal)
    ani = animation.ArtistAnimation(fig, ims, interval=0.1*1000/2, repeat=False)
    plt.axis('equal')
    plt.show(block=False)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(t,U[0,:], label='v')
    ax2.plot(t,U[1,:], label='w')
    plt.legend()
    plt.show()
    sx = np.dot(controller.S, X)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(t, sx[0,:], label='s1')
    ax3.plot(t, sx[1,:], label='s2')
    plt.legend()
    plt.show()
    # print(X)
    # print(U)
    # plt.plot(t, X[0,:])
    # plt.plot(t, X[1,:])
    # plt.plot(t, X[2,:])
    # plt.show()
    # plt.plot(X[0,:], X[1,:])
    # plt.show()