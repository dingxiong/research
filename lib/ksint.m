function aa = ksint(a0, h, nstp, d, np)
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
%   aa = ksfjaco(a0, 0.25, 1000);

if nargin < 5, np = 1; end
if nargin < 4, d = 22; end

aa = MEXksint(a0, d, h, nstp, np, 1, 0, 0, 0); 

end
