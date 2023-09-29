#!/usr/bin/env python

from math import cos, sin, pi
import numpy as np
from enum import IntEnum

class Mode(IntEnum):
    STANDBY = 0
    BACKWARD = 1
    FORWARD = 2
    FINISH = 3
    FAILED = 4
    CANCELLED = 5

class BackController():
    def __init__(self, d=0.25, D=0.5, angle_d_num=10):
        self.d = d
        self.D = D
        self.c = self.D / self.d + 1.0
        self.Kp = np.array([[1.0, 0.0],
                            [0.0, 4.0]])
        self.angle_d_num = angle_d_num
        self.angle_d_i = 0
        self.angle_d_list = []

    def calc_B12(self, phi):
        c = self.c
        sin2_phi = sin(phi)**2.0
        sin_2phi = sin(2.0*phi)
        B12 = np.array([[1.0 - c*sin2_phi,       c/2.0*sin_2phi],
                        [  c/2.0*sin_2phi, 1.0 - c + c*sin2_phi]])
        return B12

    def add_angle_d(self, angle_d):
        if len(self.angle_d_list) < self.angle_d_num:
            self.angle_d_list.append(angle_d)
        else:
            self.angle_d_list[self.angle_d_i] = angle_d
            self.angle_d_i = (self.angle_d_i + 1) % self.angle_d_num
        return self.angle_d_list

    def reset_angle_d(self):
        self.angle_d_list = []
        self.angle_d_i = 0

    def cancel(self):
        self.reset_angle_d()

    def calc_u(self, z_odom, target_pose, angle, prev_v, prev_w, mode):
        current_mode = mode
        e, phi = self.calc_e(z_odom, target_pose, angle)
        theta_d = self.calc_theta_d(e, phi)
        angle_d = calc_angle(phi - theta_d)
        angle_d_list = self.add_angle_d(angle_d)
        new_mode = self.check_mode(e, phi, angle_d_list, mode)
        u = np.zeros(2)
        if new_mode != mode:
            if not any([prev_v, prev_w]):
                current_mode = new_mode
                self.reset_angle_d()
        elif mode in [Mode.BACKWARD, Mode.FORWARD]:
            v = -0.1
            if abs(angle_d) > 30.0*pi/180.0:
                angle_d = angle_d / abs(angle_d) * 30.0*pi/180.0
            if new_mode == Mode.FORWARD:
                v = 0.1
                if sin(angle_d) > 0:
                    angle_d = -25.0*pi/180.0
                else:
                    angle_d = 25.0*pi/180.0
            if abs(sin(angle)) > 0.0001:
                r = -self.D / sin(angle)
                w = v / r
            else:
                w = 0.0
            et = (angle_d - angle) % (2.0*pi)
            if et > pi:
                et -= 2*pi
            if abs(et) > 15.0*pi/180.0:
                w = -et*0.25
                v = 0.0
            else:
                w += -et*0.25
            u = np.array([v, w])
        return u, current_mode, angle_d

    def calc_e(self, z_odom, target_pose, angle):
        z = z_odom - target_pose
        z[:2] = rotate(z[:2], -target_pose[2])
        z[2] = z[2] % (2.0*pi)
        if z[2] > pi:
            z[2] -= 2.0*pi
        phi = (z[2] + angle) % (2.0*pi)
        if phi > pi:
            phi -= 2.0*pi
        e = z[:2] - (self.d+self.D) * np.array([cos(phi), sin(phi)])
        return e, phi

    def check_mode(self, e, phi, angle_d_list, mode):
        new_mode = mode
        if abs(e[0]) < 0.1:
            if abs(e[1]) < 0.1 and abs(phi) < 10.0*pi/180.0:
                new_mode = Mode.FINISH
            else:
                new_mode = Mode.FORWARD
        elif len(angle_d_list) == self.angle_d_num:
            angle_d = sum(angle_d_list) / len(angle_d_list)
            if cos(angle_d) < cos(30.0 * pi / 180.0):
                new_mode = Mode.FORWARD
            elif cos(angle_d) > cos(10.0 * pi / 180.0) or mode == Mode.STANDBY:
                new_mode = Mode.BACKWARD
        return new_mode

    def calc_theta_d(self, e_origin, phi):
        e = np.zeros(2)
        e[:] = e_origin[:]
        if e[0] < 3.0:
            e[0] = 3.0
        v_r = - self.calc_B12(phi) @ self.Kp @ e
        vx = v_r[0]; vy = v_r[1]
        theta_d = np.arctan2(vy, vx) + pi
        return theta_d

def rotate(v, theta=-pi/2.0):
  rotateMat = np.array([
                           [cos(theta), -sin(theta)],
                           [sin(theta),  cos(theta)],
                       ])
  return np.dot(rotateMat, v)

def calc_angle(theta1, theta2=0.0):
    theta = (theta1 - theta2) % (2.0*pi)
    if theta > pi:
        theta -= 2*pi
    return theta