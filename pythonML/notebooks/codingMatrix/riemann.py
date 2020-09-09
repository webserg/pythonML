# See more information at:
# http://pyevolve.sourceforge.net/wordpress/?p=1079
# By Christian S. Perone <christian.perone@gmail.com>

# I used mencoder to create a video from all png's created
# mencoder mf://@list.txt -mf w=800:h=600:fps=%1:type=png -ovc copy -oac copy -o output.avi
# The list.txt are a list with filenames separated by line break
# The output.avi is the video output.

import mpmath
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from matplotlib.ticker import OldScalarFormatter

def calc_zeta(re, img_name):
    fig = plt.figure()
    axes = Axes3D(fig)
    axes.w_xaxis.set_major_formatter(OldScalarFormatter())
    axes.w_yaxis.set_major_formatter(OldScalarFormatter())
    axes.w_zaxis.set_major_formatter(OldScalarFormatter())

    axes.set_xlabel('X (real)')
    axes.set_ylabel('Y (imag)')
    axes.set_zlabel('Z (Zeta Img)')

    xa, ya, za  = [], [], []

    for i in np.arange(0.1, 200.0, 0.1):
        z = mpmath.zeta(complex(re, i))
        xa.append(z.real)
        ya.append(z.imag)
        za.append(i)

    axes.plot(xa, ya, za, label='Zeta Function re(s)=%.3f' % re)
    axes.legend()
    plt.grid(True)

    axes.set_xlim3d(-10.0, 12.0)
    axes.set_ylim3d(-10.0, 12.0)
    axes.set_zlim3d(0.1, 200)

    plt.savefig(img_name)
    print("Plot %s !" % img_name)
    plt.close()

if __name__ == "__main__":
    try:
        import psyco
        psyco.full()
    except ImportError:
        pass

    mpl.rcParams['legend.fontsize'] = 10
    file_index = 0
    for i in np.arange(0.01, 10.0, 0.01):
        file_index += 1
        calc_zeta(i, "zeta_plot2_%s.png" % file_index)
