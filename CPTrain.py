import math
import numpy as np
import CPENV
import CPAGNT
import matplotlib.pyplot as plt

def Angel(v1, v2):
    tmp = v1*v2
    n1 = v1*v1
    n2 = v2*v2
    return math.acos(tmp/math.sqrt(n1*n2))

def CartPoleTraining():
    theta0 = np.random.rand(1, 4)*10-5
    ita0 = np.random.rand()*2-1
    params = np.hstack((theta0, ita0))
    alpha = 0.1
    gamma = 0.95
    Delta = np.array([0, 0, 0, 0, 0])
    Delta_past = Delta
    n = 1
    eps = 3/1000

    hENV = CPENV.TCP_ENV()

    maxIter = 100000
    Vs = np.zeros(1, maxIter)
    for iIter in range(maxIter):
        delta, Vs[iIter] = CPAGNT.ExecEpi(params, gamma, hENV)
        
        Delta_past = Delta
        Delta = (n-1)/n*Delta + delta/n
        n = n + 1

        if Angel(Delta_past, Delta)<eps:
            params = param + alpha * delta
            n = 1
    
    hFile = open('result.txt', 'w')
    for i in params.size:
        print(params[i])
    hFile.close()

    plt.plot(Vs)
