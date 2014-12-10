function aa = ksint(a0, h, nstp, np, d)
% integrate KS system.
% input:
%      a0 : initial condition (must be 30x1)
%      h : time step
%      nstp: number of integration steps
%      d : size of system. Default = 22
%      np : saving spacing. Default = 1
% output:
%      aa : the orbit.  size [30, nstp/np+1]
% exmple usage:
%   aa = ksint(a0, 0.25, 1000);

if nargin < 5, d = 22; end
if nargin < 4, np = 1; end

aa = MEXksint(a0, d, h, nstp, np, 1, 0, 0, 0); 

end
