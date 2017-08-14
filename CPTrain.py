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
    elif s<-1:
        return math.pi
    else:
        return math.acos(s)

def TrainEPS(nIter, InitParams, hENV=None):
    if hENV==None:
        hENV = CPENV.TCP_ENV()
    else:
        hENV.Init()

    params = InitParams
    
    alpha = 0.1
    gamma = 0.95
    Delta = np.array([0, 0, 0, 0, 1])
    eps = 3/1000
    n = 0

    for iIter in range(nIter):
        ## perc = float(iIter) / float(nIter) * 100
        ## sys.stdout.write("%.4f%%\r" % perc)

        delta, V, t = CPAGNT.ExecEpi(params, gamma, hENV)
        
        if n==0:
            Delta = delta
            n = 1
        else:
            n = n + 1
            Delta_past = Delta
            Delta = (n-1)/n*Delta + delta/n
            if Angel(Delta_past, Delta)<eps:
                params = params + alpha * Delta
                n = 1
    n = 0
    V = 0
    t = 0
    flag = 1
    while (flag==1  and n<nIter):
        delta, V_new, t_new = CPAGNT.ExecEpi(params, gamma, hENV)
        
        if n==0:
            n = 1
            V = V_new
            t = t_new
        else:
            V_past = V
            t_past = t

            n = n + 1
            V = (n-1)/n*V + V_new/n
            t = (n-1)/n*t + t_new/n

            if abs(V-V_past)<1e-6:
                flag = 0
            if abs(t-t_past)<1e-6:
                flag = 0

    return (params, V, t)

def CPTrain():
    hENV = CPENV.TCP_ENV()

    nTrail = 20
    params_best = np.array([0, 0, 0, 0, 0])
    V_best = 0
    t_best = 0

    sys.stdout.write("Trailing for Initial Value:...\n")

    for iTrail in range(nTrail):
        perc = float(iTrail) / float(nTrail) * 100
        sys.stdout.write("\r%.4f%%" % perc)

        params = np.array([0, 0, 0, 0, 0])
        for i in range(4):
            params[i] = np.random.rand()*10-5
            params[4] = np.random.rand()*2-1

        params, V, t = TrainEPS(100, params, hENV)
        if iTrail==0:
            params_best = params
            V_best = V
            t_best = t
        else:
            if t > t_best:
                params_best = params
                V_best = V
                t_best = t
                
    params = params_best

    sys.stdout.write("\nMain Iteration:...\n")

    alpha = 0.1
    gamma = 0.95
    Delta = np.array([0, 0, 0, 0, 1])
    eps = 3/1000
    n = 0

    nIter = 100000
    Vs = np.zeros((nIter))
    ts = np.zeros((nIter))
    nUpdate = 0
    Pms = np.zeros((nIter, 5))
    for iIter in range(nIter):
        perc = float(iIter) / float(nIter) * 100
        sys.stdout.write("\r%.4f%%" % perc)

        delta, Vs[iIter], ts[iIter] = CPAGNT.ExecEpi(params, gamma, hENV)
        
        if n==0:
            Delta = delta
            n = 1
        else:
            n = n + 1
            Delta_past = Delta
            Delta = (n-1)/n*Delta + delta/n
            if Angel(Delta_past, Delta)<eps:
                params = params + alpha * Delta
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

    sys.stdout.write("\n%d\n" % nUpdate)

    ## plt.plot(Vs)

CPTrain()
