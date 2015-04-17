function [tt,aa] = ksint2M1(a0, h, T, np, d)
% integrate KS system on the 1st mode slice.
% input:
%      a0 : initial condition (must be 30x1, a0(2) = 0)
%      h : time step
%      nstp: number of integration steps
%      d : size of system. Default = 22
%      np : saving spacing. Default = 1
% output:
%      tt : time in the full state space 
%      aa : trajectory on the 1st mode slice
% exmple usage:
%   [tt,aa] = ksint2M1(a0, 0.25, 1000);

if nargin < 5, d = 22; end
if nargin < 4, np = 1; end

[tt,aa] = MEXksint(a0, d, h, 1, np, 1, 1, 1, T, 0, 0); 

end
