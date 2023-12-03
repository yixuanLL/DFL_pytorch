import torch
import numpy as np
import matplotlib.pyplot as plt

####### Kalman filter for denoise #######
# params:
# x_ : prior x
# x  : estimate x
# P  : covariance of error
# P_ : prior P
# R  : measure noise variance of x
# Q  : internal noise variance of x
# K  : Kalman gain

class KalmanFilter():
    def __init__(self, x0, Q, R):
        self.x = x0
        self.x_ = x0
        self.P = 0
        self.P_ = 0
        self.Q = Q
        self.R = R
        self.K = 0

    def predict(self):
        self.x_ = self.x
        self.P_ = self.P + self.Q
        # self.Q *= 0.6 # adaptive var
        # self.R *= 0.6 # adaptive var
        return

    def correct(self, z):
        self.K = self.P_ / (self.P_ + self.R)
        for i in range(len(z)):
             self.x[i] = self.x_[i] + self.K * (z[i] - self.x_[i])
        self.P = (1-self.K) * self.P_ #what if P is more smoothing?
        # print(self.P, self.R, self.K)
        return self.x



class KalmanFilterLayer():
    def __init__(self, x0, Q, R):
        self.x = x0
        self.x_ = x0
        self.Q = Q
        self.R = R
        self.K = 0
        self.layer_num = len(self.Q)
        self.P = [0] * self.layer_num
        self.P_ = [0] * self.layer_num
        self.K = [0] * self.layer_num

    def predict(self):
        self.x_ = self.x
        for i in range(self.layer_num):
            self.P_[i] = self.P[i] + self.Q[i]
        # self.Q *= 0.6 # adaptive var
        # self.R *= 0.6 # adaptive var
        return

    def correct(self, z):
        for i in range(self.layer_num):
            self.K[i] = self.P_[i] / (self.P_[i] + self.R[i])
            self.x[i] = self.x_[i] + self.K[i] * (z[i] - self.x_[i])
            self.P[i] = (1-self.K[i]) * self.P_[i] #what if P is more smoothing?
            # print(self.P)
        return self.x
    
# sigma = 3
# Q = 2
# R = sigma ** 2
# estimate = []
# x_list = np.array([7,4,2,5,1,8,5,5,3,8,7,6,6,6,3,6,4,1,4,7,4])
# noise = np.random.normal(loc = 0, scale=sigma, size=len(x_list))
# z_list = x_list + noise
# kfilter = KalmanFilter(z_list[0], Q, R) 

# for i in range(0, len(x_list)):
#     kfilter.predict()
#     estimate.append(kfilter.correct(z_list[i]))

# step = range(0, len(x_list))
# print(x_list)
# print(z_list)
# print(estimate)
# plt.switch_backend('agg')
# plt.plot(step, x_list, 'r', label='origin signal')
# plt.plot(step, z_list, 'g', label='noisy signal')
# plt.plot(step, estimate, 'b', label='estimation')

# plt.legend()
# plt.title('kalman filter', fontsize=9)
# plt.show()
# plt.savefig('t1.png', dpi=600)
# plt.close()