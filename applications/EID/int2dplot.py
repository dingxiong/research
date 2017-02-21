from personalFunctions import *
from py_CQCGL1d import *
from py_CQCGL2d import *

labels = ["IFRK4(3)", "IFRK5(4)",
          "ERK4(3)2(2)", "ERK4(3)3(3)", "ERK4(3)4(3)", "ERK5(4)5(4)",
          "SSPP4(3)"]
mks = ['o', 's', '+', '^', 'x', 'v', 'p']
Nscheme = len(labels)
lss = ['--', '-.', ':', '-', '-', '-', '-']

case = 10

if case == 10:
    """
    plot the integration process
    """
    N, d = 1024, 50
    Bi, Gi = 0.8, -0.6

    cgl = pyCQCGL2d(N, d, -0.1, 0.125, 0.5, 1, Bi, -0.1, Gi, 4)
    fileName = 'aa.h5'
    c2dp = CQCGL2dPlot(d, d)

    # c2dp.savePlots(cgl, fileName, 'fig1', plotType=2, size=[12, 6])
    c2dp.makeMovie('fig1')
