from personalFunctions import *
from logisticMap import Logistic

case = 20

if case == 10:
    """
    plot logistic map
    """
    logm = Logistic(5.0)
    fig, ax = logm.plotIni()
    ax2d(fig, ax)

if case == 20:
    """
    A = 5, list all periodic orbits
    """
    Eq = [
        [0, 0.8],
        [0.2535898384862274],
        [0.0575620544981097, 0.180563019141262],
        [0.0118376682323456, 0.0366524739349409, 0.2063024311119107]
    ]
    
    mp = [[-3.0, 5.0],
          [-11.0],
          [-49.4264068712, 35.4264068712],
          [ -240.961389769, 167.507590647, -103.546200878],
          [-1198.49148617, 827.861204429, 547.72002265, -481.824783312 ,-386.519433432, 313.254476004]
    ]

    logm = Logistic(5.0)
    fig, ax = logm.plotIni()
    logm.plotIter(ax, Eq[3][3], 4)
    ax2d(fig, ax)
