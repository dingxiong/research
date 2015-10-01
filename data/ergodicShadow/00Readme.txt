Author : Xiong Ding
Tue Nov 25 19:37:05 EST 2014
========================================

This folder contains the experimental data about 
exploration of dimension in KS system by use 
of Floquet vectors (FVs). Ergodic trajectories trapped
by PPO/RPOs are recorded, and the difference 
vectors are expanded by FVs. 

SO(2) symmetry is reduced by the 1st mode slice. FVs 
are projected onto the slice, so 

    Local dimension = 7  


----------------------------------------
Folder details:

case32modes: experiments with nqr = 10.
	     the po data file is "ks22h001t50x64.h5"
	     but the FVs file seems lost.
	     
case32modes_x5: experimens with nqr = 5
		the po data file is "ks22h001t120x64.h5"
	

'ppo4': distance - angle distribution data for ppo4.
	numbers of FVs used to span subspaces are:
	3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 21, 28 
	MaxT = 20000
	sT = 30;

'ppo4_old2': distance - angle distribution data for ppo4.
	numbers of FVs used to span subspaces are:
	3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 21, 28 
	MaxT = 10000
	sT = 28;

'ppo4_old1': another experiment with ppo4 but with different 
	    subspaces:
	    6 7 8 15 28
	    MaxT = 20000;
	    MaxT = 30;

'rpo1' : distance - angle distribution data for rpo1. 
       numbers of FVs used to span subspaces are:
       2, 3, 5, 6, 7, 9, 11, 13, 15, 21, 28
       MaxT = 90000
       sT = 22

'rpo3' : distance - angle distribution data for rpo1. 
       numbers of FVs used to span subspaces are:
       3, 4, 5, 7, 9, 11, 13, 15, 21, 28
       MaxT = 20000
       sT = 30

----------------------------------------
File details:
       
'angle_*': cosine of the angles, each column corresponds to a 
	   specific subspace.
'difv_*': each row represents one recorded difference vector. 
'dis_*' : the Euclidean length of each  difference vector.
'indexPo_*' : the index of closest point on PPO/RPO with the 
	    ergodic trajectory point.
'No_*' : number of close points for each approach. The summation
       of these numbers should be the number of rows in the
       corresponding 'angle_*'

disAng.mat:
	Pos: the start, end, start, end postion for shadowing part in each
	     incidence.
	Sdis: the real shadowing distance extracted from experimental data
	Sang: the real shadowing angles extracted from experimental data.

angAver.mat:
	x: the distance coordiate window.
	angleAverage: the average angle in each window
Note: for ppo4, average angles from ppo4 and ppo4_old2 are combined.

