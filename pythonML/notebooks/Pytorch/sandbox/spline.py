import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
import pylab as pl

if __name__ == '__main__':
    x = np.arange(-3, 10, 0.05)
    y = 2.5 * np.exp(-(x) ** 2 / 9) + 3.2 * np.exp(-(x - 0.5) ** 2 / 4) + np.random.normal(0.0, 1.0, len(x))
    nParam = 2
    A = np.zeros((len(x), nParam), dtype=float)
    A[:, 0] = np.exp(-(x) ** 2 / 9)
    A[:, 1] = np.exp(-(x - 0.5) ** 2 / 4)
    (p, residuals, rank, s) = np.linalg.lstsq(A, y)
    pl.plot(x, y, '.')
    pl.plot(x, p[0] * A[:, 0] + p[1] * A[:, 1], 'x', color='blue', label='Moving Average')

    spline = sig.cspline1d(y, 100)
    xbar = np.arange(-5, 15, 0.1)
    ybar = sig.cspline1d_eval(spline, xbar, dx=x[1] - x[0], x0=x[0])

    plt.plot(xbar, ybar, color='green')
    plt.legend()
    plt.show()
