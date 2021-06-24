import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# d = 0.15  # distance between wheel center and control point

class WaypointTracer():
    def __init__(self, dt=0.1) -> None:
        self.dt = dt
        self.u_pre = np.zeros(2)
        self.d = -0.15
        Kx = 1.0
        Ky = 4.0
        self.K = np.array([[Kx, 0],
                           [0, Ky]])
    
    def calc_u(self, X: np.ndarray) -> np.ndarray:
        t = X[2]
        Xcp = X[:2] + np.array([np.cos(t), np.sin(t)]) * self.d
        u = - 0.5 * np.dot(np.linalg.inv(self.calc_B(t)), np.dot(self.K, Xcp))
        u = self.limit_input(u)
        return u

    def calc_B(self, t):
        B = np.array([[np.cos(t), -self.d*np.sin(t)],
                      [np.sin(t),  self.d*np.cos(t)]])
        return B

    def limit_input(self, u):
        v_max = 0.1
        w_max = 0.3
        v = u[0]
        w = u[1]
        if abs(v) > v_max:
            w = w / abs(v) * v_max
            v = v_max * v / abs(v)
        if abs(w) > w_max:
            v = v / abs(w) * w_max
            w = w_max * w / abs(w)
        return np.array([v, w])

    def limit_acceleration(self, u):
        v_acc_max = 0.1
        w_acc_max = 0.3
        v = u[0]
        w = u[1]
        v_pre = self.u_pre[0]
        w_pre = self.u_pre[1]
        if abs(v-v_pre) > v_acc_max * self.dt:
            if v > v_pre:
                v_new = v_pre + v_acc_max * self.dt
            else:
                v_new = v_pre - v_acc_max * self.dt
            w = w * abs(v_new / v)
            v = v_new
        if abs(w-w_pre) > w_acc_max * self.dt:
            if w > w_pre:
                w_new = w_pre + w_acc_max * self.dt
            else:
                w_new = w_pre - w_acc_max * self.dt
            v = v * abs(w_new / w)
            w = w_new
        self.u_pre = np.array([v, w])
        return self.u_pre

class WaypointTracerEx(WaypointTracer):
    def calc_u(self, X: np.ndarray) -> np.ndarray:
        t = X[2]
        Xcp = X[:2] + np.array([np.cos(t), np.sin(t)]) * self.d / 1.5
        u = - 0.2 * np.dot(np.linalg.inv(self.calc_B(t)), np.dot(self.K, Xcp))
        u = self.limit_input(u)
        u = self.limit_acceleration(u)
        return u

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

def check_finish(X):
    y_tolerance = 0.05
    t_tolerance = np.pi / 36.0
    if abs(X[1]) < y_tolerance and abs(X[2] % (2*np.pi)) < t_tolerance:
        return True
    else:
        return False

if __name__ == '__main__':
    dt = 0.1
    T = 150.0
    t = np.arange(0.0, T, dt)
    t_num = len(t)
    controller = WaypointTracerEx(dt)
    X0 = np.array([2.0, 0.30, 0.0])
    X = np.zeros((3, t_num))
    X[:,0] = X0
    U = np.zeros((2, t_num))
    for i in range(t_num-1):
        U[:,i] = controller.calc_u(X[:,i])
        X[:,i+1] = update_X(X[:,i], U[:,i], dt)
        # if check_finish(X[:,i+1]):
        #     print("Break!")
        #     X = np.delete(X, np.s_[i+2:], 1)
        #     U = np.delete(U, np.s_[i+2:], 1)
        #     t = np.delete(t, np.s_[i+2:], 0)
        #     break
    theta = X[2,:]
    Xcp = X[:2,:] + np.array([np.cos(theta), np.sin(theta)]) * -0.15
    
    plt.close('all')
    tb_rect = np.array([[-0.2, 0.2, 0.2, -0.2, -0.2],
                        [-0.3, -0.3, 0.3, 0.3, -0.3]])
    area_rect = np.array([[min(X[0,:])-0.3, max(X[0,:])+0.3, max(X[0,:])+0.3],
                          [min(X[1,:])-0.3, min(X[1,:])-0.3, max(X[1,:])+0.3]])
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
            arr = plt.arrow(X[0,i], X[1,i], 0.4*np.cos(X[2,i]), 0.4*np.sin(X[2,i]), width=0.01,head_width=0.08,head_length=0.12, color='r')
            arrs = [arr]
            ims.append(area+tb+traj+goal+arrs)
    ani = animation.ArtistAnimation(fig, ims, interval=0.1*1000/2, repeat=False)
    plt.axis('equal')
    plt.show(block=False)
    print("Simulation finished!")
    print("x_error: %2.4f [m], y_error: %2.4f [m], t_error: %2.4f [deg]" % (X[0,-1], X[1,-1], X[2,-1]*180/np.pi))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(t,U[0,:], label='v')
    ax2.plot(t,U[1,:], label='w')
    ax2.plot(t,X[1,:], label='y')
    ax2.plot(t,X[2,:], label='theta')
    plt.legend()
    plt.show(block=False)
    # sx = np.dot(controller.S, X)
    # fig3 = plt.figure()
    # ax3 = fig3.add_subplot(111)
    # ax3.plot(t, sx[0,:], label='s1')
    # ax3.plot(t, sx[1,:], label='s2')
    # plt.legend()
    # plt.show()
    # print(X)
    # print(U)
    # plt.plot(t, X[0,:])
    # plt.plot(t, X[1,:])
    # plt.plot(t, X[2,:])
    # plt.show()
    # plt.plot(X[0,:], X[1,:])
    # plt.show()