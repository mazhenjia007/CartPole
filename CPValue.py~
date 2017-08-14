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

    hENV = CPENV.TCP_ENV()
    gamma = 0.95

    maxIter = 1000
    eps = 1e-6

    for i in range(row):
        perc = float(i) / float(row) * 100
        sys.stdout.write("%d / %d, %.4f%%\r" % (i, row, perc))

        V = 0
        V_past = V
        flag = 1
        params = Pms[i][:]
        n = 0

        iIter = 0
        while (flag==1 and iIter<maxIter):
            delta, V_new = CPAGNT.ExecEpi(params, gamma, hENV)
            
            sys.stdout.write("%.4f\n" % (V_new))

            if n==0:
                V_past = V
                V = V_new
                n = 1
            else:
                V_past = V
                V = (n-1)/n*V + V_new/n
                n = n + 1
            iIter = iIter + 1
            
            if abs(V - V_past) < eps:
                flag = 0

        if flag == 0:
            Vs[i] = V
        else:
            sys.stdout.write("\nError!! Iteration do not convergence!!\n")
            return

    plt.plot(Vs)
