import math
import numpy as np


def sgn(x):
    if x == 0:
        return 0
    elif x < 0:
        return -1
    else:
        return 1

class TCP_ENV:
    def __init__(self):
        ## self.s = np.array(0, 0, 0, 0)
        np.random.seed(0)
        self.s = np.random.normal(size=4) * 0.1
        self.M = 1  ## mass of cart
        self.m = 0.1  ## mass of pole
        self.l = 0.5  ## half the length of pole
        self.g = 9.8  ## gravity acceleration
        self.muc = 0.0005  ## friction coe of cart
        self.mup = 0.000002  ## friction coe of pole
        self.tau = 1 / 60  ## time interval
        self.amax = 20  ## maximum force applied
        self.a = 0  ## force applied
        self.terminal = 0  ## state of termination

    def GetState(self):
        return self.s

    def CheckTerm(self):
        if abs(self.s[0]) > 2.4:
            self.terminal = 1
            return
        if abs(self.s[2]) > 2:
            self.terminal = 1
            return
        if abs(self.s[1]) > 12 * math.pi / 180:
            self.terminal = 1
            return
        if abs(self.s[3]) > 1.5:
            self.terminal = 1
            return

    def GetReward(self):
        if self.terminal == 0:
            return 0.01
        else:
            return 0

    def SetAction(self, a_set):
        if a_set > self.amax:
            self.a = self.amax
        elif a_set < -self.amax:
            self.a = -self.amax
        else:
            self.a = a_set

    def ProcessDynamic(self):
        ay = (self.g * math.sin(self.s[1])
              + math.cos(self.s[1]) * (self.muc * sgn(self.s[2]) - self.a
                                       - self.m * self.l * self.s[3] * self.s[3] * math.sin(self.s[1]))
              / (self.M + self.m)
              - self.mup * self.s[3] / (self.m * self.l)) / (self.l
                                                             * (
                                                             0.75 - self.m * math.cos(self.s[2]) * math.cos(self.s[1]))
                                                             / (self.M + self.m))
        ax = (self.a + self.m * self.l * (self.s[3] * self.s[3] * math.sin(self.s[1])
                                          - ay * math.cos(self.s[1])
                                          - self.muc * sgn(self.s[1]))) / (self.M + self.m)
        ## print(self.s)
        self.s[0] = self.s[0] + self.s[2] * self.tau
        self.s[1] = self.s[1] + self.s[3] * self.tau
        self.s[2] = self.s[2] + ax * self.tau
        self.s[3] = self.s[3] + ay * self.tau
        self.CheckTerm()
