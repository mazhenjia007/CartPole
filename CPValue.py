import math
import numpy as np
import CPENV
import CPAGNT
import matplotlib.pyplot as plt
import sys

def CPValue():
    ## hFile = open("parameters.txt", "r")
    ## nUpdate = hFile.read()
    Pms = np.loadtxt("parameters.txt")

    row, col = Pms.shape

    Vs = np.zeros((row, 1))
    ts = np.zeros((row, 1))

    hENV = CPENV.TCP_ENV()
    gamma = 0.95

    nIter = 100
    eps = 1e-6

    for i in range(row):
        perc = float(i) / float(row) * 100
        ## sys.stdout.write("%d / %d, %.4f%%\r" % (i, row, perc))
        sys.stdout.write("\r%.4f%%" % (perc))

        V = 0
        t = 0
        flag = 1
        params = Pms[i][:]
        n = 0

        iIter = 0
        while (flag==1 and iIter<nIter):
            delta, V_new, t_new = CPAGNT.ExecEpi(params, gamma, hENV)
            if n==0:
                V = V_new
                t = t_new
                n = 1
            else:
                n = n + 1
                V_past = V
                t_past = t
                V = (n-1)/n*V + V_new/n
                t = (n-1)/n*t + t_new/n
                if abs(V - V_past) < eps:
                    flag = 0
                if abs(t - t_past) < eps:
                    flag = 0

            iIter = iIter + 1

        if flag == 0:
            Vs[i] = V
            ts[i] = t
        else:
            sys.stdout.write("\nError!! Iteration do not convergence!!\n")
            return

    plt.plot(ts)

CPValue()
