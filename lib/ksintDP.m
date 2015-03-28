function [aa, D, P] = ksintDP(a0, h, nstp, np, d)
% integrate KS system.
% input:
%      a0 : initial condition (must be 30x1)
%      h : time step
%      nstp: number of integration steps
%      d : size of system. Default = 22
%      np : saving spacing. Default = 1
% output:
%      aa : the orbit.  size [30, nstp/np+1]
%      D  : dissipation along the orbit. D(0) = 0. size [nstp/np+1,1]
%      P  : pump along the orbit. P(0) = 0. size [nstp/np+1,1]
% exmple usage:
%   [aa, D, P] = ksint(a0, 0.25, 1000);

if nargin < 5, d = 22; end
if nargin < 4, np = 1; end

[aa, D, P] = MEXksint(a0, d, h, nstp, np, 1, 0, 0, 0, 1); 

end
