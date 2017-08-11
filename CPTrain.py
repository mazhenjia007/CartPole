import math
import numpy as np
import CPENV
import CPAGNT
import matplotlib.pyplot as plt
import sys

def Angel(v1, v2):
    tmp = v1*v2
    n1 = v1*v1
    n2 = v2*v2

    inp = tmp.sum()
    l1 = n1.sum()
    l2 = n2.sum()

    ## print("%.4f %.4f %.4f" % (inp, math.sqrt(l1), math.sqrt(l2)))
    ## print(inp/math.sqrt(l1*l2))

    s = inp/math.sqrt(l1*l2)
    if s>1:
        return 0
    else:
        return math.acos(inp/math.sqrt(l1*l2))

def CPTrain():
    ## theta0 = np.random.rand(4, 1)*10-5
    ## ita0 = np.random.rand(1, 1)*2-1
    ## params = np.hstack((theta0, ita0))
    params = np.array([0, 0, 0, 0, 0])
    for i in range(4):
        params[i] = np.random.rand()*10-5
    params[4] = np.random.rand()*2-1

    alpha = 0.1
    gamma = 0.95
    Delta = np.array([1, 1, 1, 1, 1])
    ## Delta_past = Delta
    n = 0
    eps = 3/1000

    hENV = CPENV.TCP_ENV()

    maxIter = 100000
    Vs = np.zeros((maxIter))
    nUpdate = 0
    Pms = np.zeros((maxIter, 5))
    for iIter in range(maxIter):
        perc = float(iIter) / float(maxIter) * 100
        ## sys.stdout.write("%d / %d, %.4f%%\r" % (iIter, maxIter, perc))
        sys.stdout.write("%.4f%%\r" % perc)

        delta, Vs[iIter] = CPAGNT.ExecEpi(params, gamma, hENV)
        
        if n==0:
            Delta_past = Delta
            Delta = delta
            n = 1
        else:
            Delta_past = Delta
            Delta = (n-1)/n*Delta + delta/n
            n = n + 1

        if Angel(Delta_past, Delta)<eps:
            params = params + alpha * delta
            n = 1
            Pms[nUpdate, :] = params
            nUpdate = nUpdate + 1
    
    hFile = open('result.txt', 'w')
    hFile.write("%f %f %f %f %f\n" % (params[0], params[1], params[2], params[3], params[4]))
    hFile.close()

    hFile = open('parameters.txt', 'w')
    ## hFile.write("%d\n" % (nUpdate))
    for i in range(nUpdate):
        for j in range(5):
            hFile.write("%f " % (Pms[i][j]))
        hFile.write("\n")
    hFile.close()

    print(nUpdate)

    ## plt.plot(Vs)
