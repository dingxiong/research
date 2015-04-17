function [tt,aa] = ksintM1(a0, h, nstp, np, d)
% integrate KS system on the 1st mode slice.
% input:
%      a0 : initial condition (must be 30x1, a0(2) = 0)
%      h : time step
%      nstp: number of integration steps
%      d : size of system. Default = 22
%      np : saving spacing. Default = 1
% output:
%    tt : time sequence in the full state space [nstp/np+1]
%    aa : trajectory in the 1st mode slice [30, nstp/np+1]
% exmple usage:
%   [tt, aa] = ksintM1(a0, 0.25, 1000);

if nargin < 5, d = 22; end
if nargin < 4, np = 1; end

[tt,aa] = MEXksint(a0, d, h, nstp, np, 1, 1, 0, 0, 0, 0); 

end
