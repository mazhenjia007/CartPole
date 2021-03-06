import math
import numpy as np
import sys

def sgn(x):
    if x == 0:
        return 0
    elif x < 0:
        return -1
    else:
        return 1

class TCP_ENV:
    def __init__(self):
        ## np.random.seed(0)
        self.s = np.zeros(4)
        self.Init()
        self.M = 1  ## mass of cart
        self.m = 0.1  ## mass of pole
        self.l = 0.5  ## half the length of pole
        self.g = 9.8  ## gravity acceleration
        self.muc = 0.0005  ## friction coe of cart
        self.mup = 0.000002  ## friction coe of pole
        self.tau = 1 / 60  ## time interval
        self.amax = 20  ## maximum force applied
        self.Q = np.array([1.25, 1, 12, 0.25]) ## reward coe1
        self.R = 0.01 ## reward coe2
        self.time = 0 ## time step
        self.maxtime = 100 ## over this time, regarded as convergence

    def Init(self):
        ## self.s = np.random.normal(size=4) * 0.1
        self.s[0] = np.random.normal(0, 0.1)
        self.s[1] = np.random.normal(0, 0.1)
        self.s[2] = np.random.normal(0, 0.03)
        self.s[3] = np.random.normal(0, 0.1)
        self.terminal = 0  ## state of termination
        self.a = 0 ## force applied
        self.time = 0

    def GetState(self):
        return self.s

    def CheckTerm(self):
        if abs(self.s[0]) > 2.4:
            self.terminal = 1
            return
        if abs(self.s[1]) > 2:
            self.terminal = 1
            return
        if abs(self.s[2]) > 12 * math.pi / 180:
            self.terminal = 1
            return
        if abs(self.s[3]) > 1.5:
            self.terminal = 1
            return
        if self.time > self.maxtime:
            self.terminal = -1
            return

    def GetReward(self):
        tmp = self.s*self.Q*self.s
        r = -tmp.sum() - self.a*self.R*self.a
        return r

    def SetAction(self, a_set):
        if a_set > self.amax:
            self.a = self.amax
        elif a_set < -self.amax:
            self.a = -self.amax
        else:
            self.a = a_set

    def ProcessDynamic(self):
        vx = self.s[1]
        y = self.s[2]
        vy = self.s[3]
        sy = math.sin(y)
        cy = math.cos(y)
        sgnx = sgn(vx)
        ay = (self.g*sy + cy*(self.muc*sgnx - self.a - self.m*self.l*(vy**2)*sy)/(self.M + self.m)
              - self.mup*vy/(self.m*self.l)) / (self.l*(0.75 - self.m*(cy**2)/(self.M + self.m)))
        ax = (self.a + self.m*self.l*((vy**2)*sy - ay*cy) - self.muc*sgnx)/(self.M + self.m)
        ## print(self.s)
        self.s[0] = self.s[0] + self.s[1] * self.tau
        self.s[1] = self.s[1] + ax * self.tau
        self.s[2] = self.s[2] + self.s[3] * self.tau
        self.s[3] = self.s[3] + ay * self.tau
        self.time = self.time + self.tau
        self.CheckTerm()

    def Display(self):
        x = self.s[0]
        vx = self.s[1]
        y = self.s[2]
        vy = self.s[3]
        sys.stdout.write("Time:%5.3f: a=%5.2f x=%5.2f vx=%5.2f y=%5.2f vy=%5.2f\n" 
                         % (self.time, self.a, x, vx, y, vy))
