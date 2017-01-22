from personalFunctions import *

case = 10

if case == 10:
    """
    check the angle along one orbit
    maxElement.dat stores the data for ppo147 the largest absolute 
    off diagonal elements
    """
    f = './anglePOs64/ppo/space/147/ang1.dat'
    ang = np.arccos(np.loadtxt(f))

    fig, ax = pl2d(yscale='log', labs=[r'$t$', r'$\theta$'], tickSize=18)
    ax.plot(ang[::10])
    ax2d(fig, ax)

    off = np.loadtxt('maxElement.dat')
    fig, ax = pl2d(yscale='log', labs=[r'$t$', r'$\max|R_{ij}|$'], tickSize=18)
    ax.plot(off)
    ax2d(fig, ax)
    

    
    
