import h5py
import numpy as np
from numpy.linalg import norm, qr, svd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from IPython.display import display
# from IPython.html.widgets import interact
from py_ks import *


def pAngle(A, B):
    A = A/norm(A)
    A.resize((np.size(A), 1))
    U, s, V = svd(np.dot(A.T, B))
    # print s
    # s is 1-d array
    return np.arccos(s[0])


def seqAng(seqDifv, ve):
    # seqDifv is a sequence of row vector
    M, N = seqDifv.shape
    ang = np.zeros(M)
    for i in range(M):
        ang[i] = pAngle(seqDifv[i, :], ve)
    return ang


def calMinDis(prePo, preDifv, nowPo, ksm1, t):
    a0Ergo = prePo + preDifv
    aaErgo, phase = ksm1.intg2(a0Ergo, t, 1)
    N, M = aaErgo.shape
    newDifv = aaErgo - nowPo
    difNorm = norm(newDifv, axis=1)
    newMinDis = np.amin(difNorm)
    minIndex = np.argmin(difNorm)
    return newDifv, difNorm, newMinDis, minIndex

##################################################
# load data
ppType = 'ppo'
ppId = 4
nsp = 5
gTpos = 3
case = 'case4ppo4x5sT30'
ang = np.loadtxt(case+'/angle')
ang = np.sin(np.arccos(ang))
dis = np.loadtxt(case+'/dis')
difv = np.loadtxt(case+'/difv')
No = np.loadtxt(case+'/No')
indexPo = np.loadtxt(case+'/indexPo')

f = h5py.File('../../data/ks22h001t120x64EV.h5', 'r')
dataName = '/' + ppType + '/' + str(ppId) + '/'
a0 = f[dataName + 'a'].value
T = f[dataName + 'T'].value[0]
nstp = np.int(f[dataName + 'nstp'].value[0])
h = T / nstp
e = f[dataName + 'e'].value
veAll = f[dataName + 've'].value

ks = pyKS(64, h, 22)
aa = ks.intg(a0, nstp, 5)
aa = aa[:-1, :]
if ppType == 'ppo':
    aaWhole = ks.half2whole(aa)
    veWhole = ks.half2whole(veAll)
else:
    aaWhole = aa
    veWhole = veAll

##################################################
# plot the orbit and the total distribution

aaHat = ks.orbitToSlice(aaWhole)[0]
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(aaHat[:, 0], aaHat[:, 2], aaHat[:, 3])
plt.show(block=False)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.scatter(dis, ang[:, 4], s=7, c='r', marker='o',
           edgecolor='none', label='1-'+str(7))
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([1e-4, 1e-1])
ax.set_xlim([1e-4, 1e-1])
ax.legend(loc='upper left')
plt.show(block=False)

##################################################
# problematic incidence

# ix = 3;
# ix = 155;
ix = 104
x1 = np.int(np.sum(No[:ix]))
x2 = np.int(np.sum(No[:ix+1]))
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.scatter(dis[x1:x2], ang[x1:x2, 4], s=7, c='r', marker='o',
           edgecolor='none', label='1-'+str(7))
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([1e-4, 1e-1])
ax.set_xlim([1e-4, 1e-1])
ax.legend(loc='upper left')
plt.show(block=False)

# one single point
minDis = np.amin(dis[x1:x2])
pid = np.argmin(dis[x1:x2])
print minDis, ang[x1+pid, 4], indexPo[x1+pid], norm(difv[x1+pid, :])

##################################################

pointPo = aaHat[indexPo[x1+pid], :]
a0Ergo = pointPo + difv[x1+pid, :]  # recover the ergodic point
dif = a0Ergo - aaHat[indexPo[x1+pid]-10:indexPo[x1+pid]+11, :]
nor = norm(dif, axis=1)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(nor, '-o')
# ax.set_yscale('log')
# ax.set_xscale('log')
plt.show(block=False)
print np.amin(nor)


##################################################
# use smaller time step to integrate the ergodic trajectory

ksm1 = pyKSM1(64, 0.001, 22)
prePo = aaHat[indexPo[x1+pid-1]]
preDifv = difv[x1+pid-1]
nowPo = aaHat[indexPo[x1+pid]]
newDifv, difNorm, newMinDis, minIndex = calMinDis(prePo, preDifv, nowPo, ksm1, 0.2)

# plot the new distance
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(difNorm)
# ax.set_yscale('log')
# ax.set_xscale('log')
plt.show(block=False)
print np.amin(difNorm)

ve = veWhole[indexPo[x1+pid], :].reshape([30, 62])
ve = ks.veToSlice(ve, aaWhole[indexPo[x1+pid], :])  # transform to slice
# print ve[gTpos,:].T
# get rid of one tangent direction
FV = np.vstack((ve[0:gTpos, :], ve[gTpos+1:, :]))
# print ve.shape
ve = qr(FV.T)[0]
# print ve[:, 0:3]

##################################################
# plot the new angle and the old angle in the same plot

oldDifv = difv[x1+pid, :]
trueDifv = newDifv[minIndex, :]
oldAngle = np.sin(pAngle(oldDifv, ve[:, 0:7]))
newAngle = np.sin(pAngle(trueDifv, ve[:, 0:7]))

# print out the original angle and new angle
print oldAngle, newAngle
# plot the old point and new point
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.scatter(dis[x1:x2], ang[x1:x2, 4], s=7, c='r', marker='o',
           edgecolor='none', label='1-'+str(7))
ax.scatter(dis[x1+pid], ang[x1+pid, 4], s=30, c='k', marker='s', label='old')
ax.scatter(newMinDis, newAngle, s=30, c='b', marker='^', label='new')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([1e-4, 1e-1])
ax.set_xlim([1e-4, 1e-1])
ax.legend(loc='upper left')
plt.show(block=False)


##################################################
# investigate how the angle changes.
#
# There is a little blow-up at the closest point.
# If there is no such blow-up, then data looks perfect,
# On the other hand, this blow-up looks reasonable because
# difference vector is perpendicular to the velocity field
# at the closest point; while at far-away point, velocity
# , as one FVs, make an important role in spanning difference
# vector.

tmpAngs = seqAng(newDifv, ve[:, 0:7])

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(tmpAngs)
ax.set_yscale('log')
# ax.legend(loc='upper left')
plt.tight_layout(pad=0)
plt.show(block=False)

##################################################
# have a look at how well the new point lies on the Poincare
# section defined by velocity
# $$ \frac{\Delta x \cdot v}{|\Delta x|\cdot |v|} $$
# Then try to interpolate the points to get an exact point on the
# Poincare section, but the angle nearly changes.

velocity = ks.velocity(aaHat[indexPo[x1+pid], :])
velocity = ks.veToSlice(velocity, aaHat[indexPo[x1+pid], :]).squeeze()
dotproduct = np.dot(newDifv, velocity)
Poincare = dotproduct / norm(velocity) / norm(newDifv, axis=1)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(Poincare)
# ax.set_yscale('log')
# ax.legend(loc='upper left')
plt.tight_layout(pad=0)
plt.show(block=False)

print Poincare[minIndex], np.amin(np.abs(Poincare))

##################################################
# interpolate
interpDifv = -dotproduct[minIndex+1] * newDifv[minIndex] + dotproduct[minIndex] * newDifv[minIndex+1]
print np.dot(interpDifv, velocity) / norm(velocity) / norm(interpDifv)
print np.sin(pAngle(interpDifv, ve[:, 0:7]))


##################################################
# plot the angle between difference vector and single FV when
# ergodic point slides along the ergodic trajectory

ixFV = np.array([0, 1, 7, 10])
fig = plt.figure(figsize=(8, 6))
for i in range(ixFV.size):
    angDxFV = np.dot(newDifv, FV[ixFV[i]]) / norm(FV[ixFV[i]]) / norm(newDifv, axis=1)
    ax = fig.add_subplot('22'+str(i+1))
    ax.plot(angDxFV)
    ax.plot([minIndex, minIndex], [-1, 1], c='r', lw=2, label=str(ixFV[i]))
    ax.grid('on')
    ax.legend(loc='upper left')
plt.tight_layout(pad=0)
plt.show(block=False)

##################################################
# apply this method to all points in this case
# this section is put at the end to avoid namespace polution

newCaseAngle = np.zeros(x2-x1)
newCaseDis = np.zeros(x2-x1)
for i in range(x1+1, x2+1):
    prePo = aaHat[indexPo[i-1]]
    preDifv = difv[i-1]
    nowPo = aaHat[indexPo[i]]
    newDifv, difNorm, newMinDis, minIndex = calMinDis(prePo, preDifv, nowPo, ksm1, 0.2)
    ve = veWhole[indexPo[i]].reshape([30, 62])
    ve = ks.veToSlice(ve, aaWhole[indexPo[i], :])
    ve = np.vstack((ve[0:gTpos, :], ve[gTpos+1:, :]))
    ve = qr(ve.T)[0]
    newAngle = np.sin(pAngle(newDifv[minIndex], ve[:, 0:7]))
    newCaseAngle[i-(x1+1)] = newAngle
    newCaseDis[i-(x1+1)] = difNorm[minIndex]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.scatter(newCaseDis, newCaseAngle, s=7, c='r', marker='o',
           edgecolor='none', label='1-'+str(7))
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([1e-4, 1e-1])
ax.set_xlim([1e-4, 1e-1])
ax.legend(loc='upper left')
plt.show(block=False)
