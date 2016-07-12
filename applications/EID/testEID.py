from personalFunctions import *

case = 20

if case == 10:
    aas = []
    ltes = []
    for i in range(6):
        aa = np.loadtxt('data/aa' + str(i)+'.dat')
        lte = np.loadtxt('data/lte' + str(i)+'.dat')
        aas.append(aa.T)
        ltes.append(lte)
        
    i = 0
    plot3dfig(aas[i][:, 0], aas[i][:, 1], aas[i][:, 2])
    
    labels = ["Cox_Matthews", "Krogstad", "Hochbruck_Ostermann",
              "Luan_Ostermann", "IFRK43", "IFRK54"]
    T = 10.25
    fig, ax = pl2d(labs=[r'$t$', r'$LTE$'], yscale='log', xlim=[0, 2*T])
    for i in range(6):
        n = len(ltes[i])
        ax.plot(np.linspace(0, 2*T, n), ltes[i], lw=2, label=labels[i])
    ax2d(fig, ax)
        
if case == 20:
    ltes = []
    for i in range(3):
        k = 10**i
        lte = np.loadtxt('N20_lte' + str(k) + '.dat')
        ltes.append(lte)
    T = 10.25
    fig, ax = pl2d(labs=[r'$t$', r'$LTE$'], yscale='log', xlim=[0, 2*T])
    k = 5
    for i in range(3):
        n = len(ltes[i][:, k])
        ax.plot(np.linspace(0, 2*T, n), ltes[i][:, k], lw=2)
    ax2d(fig, ax)
