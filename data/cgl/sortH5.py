from personalFunctions import *

case = 20

if case == 10:
    """
    allocate all soliton solutions
    """
    ix = np.arange(0.09, 0.57, 0.01).tolist()
    ixf = [format(i, ".6f") for i in ix]
    fs = ['req09.h5', 'req1.h5', 'req11.h5', 'req12.h5', 'req13.h5',
          'req14.h5', 'req15.h5', 'req16.h5', 'req17.h5', 'req18.h5',
          'req19.h5', 'req2.h5', 'req21.h5', 'req22.h5', 'req23.h5',
          'req24.h5', 'req25.h5', 'req26.h5', 'req27.h5', 'req28.h5',
          'req29.h5', 'req3.h5', 'req31.h5', 'req32.h5', 'req33.h5',
          'req34.h5', 'req35.h5', 'req36.h5', 'req37.h5', 'req38.h5',
          'req39.h5', 'req4.h5', 'req41.h5', 'req42.h5', 'req43.h5',
          'req44.h5', 'req45.h5', 'req46.h5', 'req47.h5', 'req48.h5',
          'req49.h5', 'req5.h5', 'req51.h5', 'req52.h5', 'req53.h5',
          'req54.h5', 'req55.h5', 'req56.h5']
    inps = ['./req_different_di/' + i for i in fs]
    for i in range(len(ix)):
        cqcglMoveReqEV(inps[i], '1', 'reqDi.h5', ixf[i]+'/1')

if case == 20:
    """
    allocate all soliton solutions
    """
    ix = [0.57, 0.59, 0.61, 0.63, 0.65, 0.67, 0.69, 0.71, 0.74, 0.77,
          0.8, 0.83, 0.86, 0.9, 0.94, 0.96]
    ixf = [format(i, ".6f") for i in ix]
    fs = ['req57.h5', 'req59.h5', 'req61.h5', 'req63.h5', 'req65.h5',
          'req67.h5', 'req69.h5', 'req71.h5', 'req74.h5', 'req77.h5',
          'req8.h5', 'req83.h5', 'req86.h5', 'req9.h5', 'req94.h5',
          'req96.h5']
    inps = ['./req_different_di/' + i for i in fs]
    for i in range(len(ix)):
        cqcglMoveReqEV(inps[i], '1', 'reqDi.h5', ixf[i]+'/1')
