import math
import numpy as np
import CPENV
from scipy.stats import norm

def ExecEpi(params, gamma, hENV=None):
    if hENV==None:
        hENV = CPENV.TCP_ENV()
    else:
        hENV.Init()

    ## init
    theta = params[0:4]
    ita = params[4]
    z = np.array([0, 0, 0, 0, 0])
    delta = np.array([0, 0, 0, 0, 0])
    t = 0

    C = np.array([1/2.4, 1/2, 180/(12*math.pi), 1/1.5])
    V = 0

    while hENV.terminal==0:
        s = hENV.GetState()
        ## draw an action a
        tmp = theta*C*s
        mu = tmp.sum()
        sigma = 0.1 + 1/(1+math.exp(ita))
        a = np.random.normal()*sigma+mu
        ## p = norm.pdf(a, mu, sigma)

        ## take a and get reward
        hENV.SetAction(a)
        hENV.ProcessDynamic()
        r = hENV.GetReward()

        ## calculate parameters
        d1 = (a-mu)/(sigma**2)*(C*s)
        ## d2 = math.exp(ita)/(sigma*((1+math.exp(ita))**2))*(1-((a-mu)/sigma)**2)
        d2 = (1-(((a-mu)/sigma)**2))/sigma*math.exp(ita)/((1+math.exp(ita))**2)
        z = z + np.hstack((d1, d2))
        delta = delta + (r*(gamma**t))*z
        V = V + r*(gamma**t)
        t = t + 1

    return (delta, V)
