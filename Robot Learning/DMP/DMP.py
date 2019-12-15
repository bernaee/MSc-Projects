import numpy as np


class DMP():
    def __init__(self, x, vel, acc, time, K, D, alpha, tau):
        self.x = x
        self.vel = vel
        self.acc = acc
        self.time = time
        self.K = K
        self.D = D
        self.alpha = alpha
        self.tau = tau
        self.data_length = time.shape[0]
        self.ts = time[-1] / time.shape[0]

    def set_initial_pos(self, x_0):
        self.x_0 = x_0

    def set_goal_pos(self, g):
        self.g = g

    def set_x(self, x):
        self.x = x

    def set_time(self, time):
        self.time = time

    def set_tau(self, tau):
        self.tau = tau

    def set_s(self):
        self.s = np.exp(-(self.alpha / self.tau) * self.time)

    def set_basis_functions(self, bf_number):
        self.bf_number = bf_number
        self.c = np.logspace(-3, 0, num=self.bf_number)
        self.h = self.bf_number / (self.c ** 2)

    def generate_psi(self):
        psi = np.zeros((self.data_length, self.bf_number))
        for i in range(self.bf_number):
            psi[:, i] = np.exp(-1 * self.h[i] * ((self.s - self.c[i]) ** 2))
        self.psi = psi

    def set_f_target(self):
        self.f_target = (-self.K * (self.g - self.x) + self.D * self.vel + self.tau * self.acc) / (self.g - self.x_0)

    def set_f(self):
        self.y = np.multiply(self.f_target, self.psi.sum(axis=1)) / self.s
        self.weights = np.dot(np.linalg.inv(np.dot(self.psi.T, self.psi)), np.dot(self.psi.T, self.y))
        canonical_f = np.zeros((self.data_length, self.bf_number))
        for i in range(self.bf_number):
            canonical_f[:, i] = np.multiply(self.psi[:, i], self.weights[i]) / self.psi.sum(axis=1)
            canonical_f[:, i] = np.multiply(canonical_f[:, i], self.s)
        self.f = canonical_f.sum(axis=1)
        self.cost = np.power((self.f - self.f_target), 2).sum()

    def reproduce_movement(self, perturbation=False):
        x_new = self.x_0
        vel_new = 0
        self.movement = np.zeros((self.data_length, 3))
        for index in range(self.data_length):
            acc_new = (self.K * (self.g - x_new) - self.D * vel_new + (self.g - self.x_0) * self.f[
                index])
            vel_new = (vel_new + acc_new / self.tau)
            x_new = x_new + vel_new

            self.movement[index, 0] = x_new
            self.movement[index, 1] = vel_new
            self.movement[index, 2] = acc_new

            if bool(perturbation):
                if perturbation['index'] == index:
                    x_new = x_new + perturbation['amount']
