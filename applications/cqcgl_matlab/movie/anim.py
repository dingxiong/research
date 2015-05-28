import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

N = 1024; M = 10000;
AA = np.fromfile('aa.bin', np.double, 2*N*M).reshape(M,2*N).T;
Ar = AA[0::2,:]; Ai = AA[1::2, :]; Ama = abs(Ar+1j*Ai);

fig = plt.figure(figsize=(8,6));
ax = fig.add_subplot(111)
ax.set_ylim((-0.5, 3.5))
ax.set_xlabel('L'); 
ax.set_ylabel('|A|');
ax.grid('on', color ='w', lw = 1); fig.tight_layout(pad = 0)
ax.patch.set_facecolor('black')

x = np.linspace(0, 50, N, endpoint=False)
line, = ax.plot(x, Ama[:,0], c=(0,1,0) ,lw=1.5, label = '0');

Maxf = 7000 ;
def animate(i):
    y = Ama[:,i];
    line.set_ydata(y);
    # only use the integer part to get rid of flashes of legend.
    line.set_label('time : ' + str( int((i*0.005+500)*10)/10.0 ) ); 
    ax.legend(fancybox=True)
    return line, ax; # ax also needs be updated in order to update legend.

anim = animation.FuncAnimation(fig, animate, frames=Maxf,
                               interval=0 , blit=False, repeat=False);
anim.save('cqcgl1d.mp4', dpi=100, fps=30, extra_args=['-vcodec', 'libx264'])
#plt.show()
