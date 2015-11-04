import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import *
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
from personalFunctions import *

##################################################


def add_subplot_axes(ax, rect, axisbg='w'):
    """
    function to add embedded plot
    rect = [ x of left corner, y of left corner, width, height ] in
    percentage
    """
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def setAxis(ax):
    # ax.set_xlabel(r'$\theta$', fontsize=20, labelpad=-40)
    ax.set_xticks([0., .125*pi, 0.25*pi, 0.375*pi, 0.5*pi])
    ax.set_xticklabels(["$0$", r"$\frac{1}{8}\pi$", r"$\frac{1}{4}\pi$",
                        r"$\frac{3}{8}\pi$", r"$\frac{1}{2}\pi$"],
                       fontsize=15)
    # ax.legend(loc='best', fontsize=12)
    ax.set_yscale('log')
    ax.set_ylim([0.01, 100])
    ax.text(0.02, 20, r'$\rho(\theta)$', fontsize=20)
    # ax.set_ylabel(r'$\rho(\theta)$', fontsize=20, labelpad=-55)


def filterAng(a, ns, angSpan, angNum):
    """
    filter the bad statistic points
    because a very small fraction of Floquet vectors corresponding
    to high Fourier modes are not well told apart, so there are
    a small fraction of misleading close to zeros angles for
    non-physical space.
    """
    for i in range(29):
        n = a[0].shape[0]
        for j in range(n):
            y = a[i][j]*ns/(angSpan[i]*angNum[i])
            if y < 2e-2:
                a[i][j] = 0
    
##################################################
# load data
ixRange = range(0, 29)
N = len(ixRange)

folder = './anglePOs64/ppo/space/'
fileName = [folder + 'ang' + str(i) + '.dat' for i in ixRange]
folder2 = './anglePOs64/rpo/space/'
fileName2 = [folder2 + 'ang' + str(i) + '.dat' for i in ixRange]
ns = 1000

# collect the data with rpo, ppo combined
angs = []
for i in range(N):
    print i
    ang = arccos(loadtxt(fileName[i]))
    ang2 = arccos(loadtxt(fileName2[i]))
    angs.append(np.append(ang, ang2))
    # angs.append(ang2)
##################################################

case = 1

if case == 1:
    """
    angle distribution in ppo and rpo
    """
    a = []
    b = []
    angNum = []
    angSpan = []
    for i in range(N):
        print i
        angNum.append(angs[i].shape[0])
        angSpan.append(max(angs[i])-min(angs[i]))
        at, bt = histogram(angs[i], ns)
        a.append(at)
        
    b = bt
    filterAng(a, ns, angSpan, angNum)
    labs = ['k='+str(i+1) for i in ixRange]
    
    
    np.savez_compressed('ab', a=a, b=b, ns=ns, angSpan=angSpan,
                        angNum=angNum)

    fig = plt.figure(figsize=(4, 1.5))
    ax = fig.add_subplot(111)
    for i in range(7):
        ax.plot(b[:-1], a[i]*ns/(angSpan[i]*angNum[i]), label=labs[i], lw=1.5)
    setAxis(ax)
    plt.tight_layout(pad=0)
    plt.show()

    fig = plt.figure(figsize=(4, 1.5))
    ax = fig.add_subplot(111)
    colors = cm.rainbow(linspace(0, 1, 11))
    for ix in range(11):
        i = 7 + 2*ix
        ax.plot(b[:-1], a[i]*ns/(angSpan[i]*angNum[i]), c=colors[ix], lw=1.5)
    setAxis(ax)
    plt.tight_layout(pad=0)
    plt.show()
    
if case == 2:
    a = []
    b = []
    angNum = []
    angSpan = []
    for i in range(N):
        print i
        angNum.append(angs[i].shape[0])
        angSpan.append(max(angs[i])-min(angs[i]))
        at, bt = histogram(log(angs[i]), ns)
        a.append(at)
        b.append(bt)
    
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    colors = cm.rainbow(linspace(0, 1, N))
    #labs = ['(1-' + str(i+1) + ',' + str(i+2) + '-30)' for i in ixRange]
    labs = ['(' + str(i+1) + ',' + str(i+2) + ')' for i in ixRange]
    labx =[pi/16, pi/8*1.1, pi/8*0.9, pi/8*3*0.9, pi/2*0.8]
    laby =[0.9, 0.02, 0.1, 0.05, 0.005]


    ax.set_xlabel(r'$\theta$',size='large')
    ax.set_xticks([0., .125*pi, 0.25*pi, 0.375*pi])
    ax.set_xticklabels(["$0$", r"$\frac{1}{8}\pi$", r"$\frac{2}{8}\pi$",
                        r"$\frac{3}{8}\pi$"])

    for i in range(9):
        ax.scatter(b[i][:-1], a[i]*ns/(angSpan[i]*angNum[i]), s = 7, 
                   c = colors[i], edgecolor='none', label=labs[i])

    ax.legend(fontsize='small', loc='upper center', ncol = 1,
              bbox_to_anchor=(1.1, 1), fancybox=True)
    ax.set_yscale('log')
    ax.set_ylim([0.001, 1000])
    ax.set_ylabel(r'$\rho(\theta)$',size='large')


    plt.tight_layout(pad=0)
    plt.show()


if case == 3:
    nstps = []
    for i in range(200):
        a, T, nstp, r, s = KSreadPO('../ks22h001t120x64EV.h5', 'rpo', i+1)
        nstps.append(nstp)
    nstps = np.array(nstps)
