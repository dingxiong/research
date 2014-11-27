import matplotlib.pyplot as plt
from numpy import *
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D

##################################################
# function to add embedded plot
# rect = [ x of left corner, y of left corner, width, height ] in 
# percentage
def add_subplot_axes(ax,rect,axisbg='w'):
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
##################################################

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot(111)
ns = 1000;
ReverseAngle = False;

fileName = ['./ppo/ang_55_66.txt', './ppo/ang_67_89.txt',
            './ppo/ang_88_99.txt', './ppo/ang_1011_1213.txt',
            './ppo/ang_1415_2627.txt'];
colors = ['r', 'b', 'c', 'm', 'DarkGreen'];
labs =['(6,7)', '(7-8,9-10)', '(9,10)', '(11-12,13-14)', '(15-16,27-28)']
labx =[pi/16, pi/8*1.1, pi/8*0.9, pi/8*3*0.9, pi/2*0.8]
laby =[0.9, 0.02, 0.1, 0.05, 0.005]

N = len(fileName); ang =[]; a = []; b = [];
for i in range(N):
    ang.append(arccos(loadtxt(fileName[i])));
    at, bt = histogram(ang[i], ns);
    a.append(at); b.append(bt);

if(ReverseAngle):
    for i in range(N):
        b[i] = 1/b[i];
    ax2 = add_subplot_axes(ax, [0.6, 0.1, 0.4, 0.5])
    ax2.set_xlim(0,200)
    ax.set_xlabel(r'$1/\theta$',size='large')
else:
    ax.set_xlabel(r'$\theta$',size='large')
    ax.set_xticks([0., .125*pi, 0.25*pi, 0.375*pi])
    ax.set_xticklabels(["$0$", r"$\frac{1}{8}\pi$", r"$\frac{2}{8}\pi$",
                        r"$\frac{3}{8}\pi$"])
########################################

for i in range(N):
    ax.scatter(b[i][:-1], a[i]*ns/max(ang[i])/ang[i].shape[0], s = 7, 
               c = colors[i], edgecolor='none')
    if(not ReverseAngle):
        ax.text(labx[i], laby[i], labs[i])

ax.set_yscale('log')
#ax.set_title('(a)')
ax.set_ylabel(r'$\rho(\theta)$',size='large')

if(ReverseAngle):
    ax2.scatter(b[0][:-1], a[0]*ns/max(ang[0])/ang[0].shape[0], s = 7, 
               c = colors[0], edgecolor='none')
    #ax2.scatter(b[2][:-1], a[2]*ns/max(ang[2])/ang[2].shape[0], s = 7, 
    #           c = colors[2], edgecolor='none')
    ax2.set_yscale('log')
##############################################

plt.tight_layout(pad=0)
plt.show()
