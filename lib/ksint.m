function [aa, daa] = ksint(a0, h, nstp, np, nqr, d)
% integrate KS system.
% input:
%      a0 : initial condition (must be 30x1)
%      h : time step
%      nstp: number of integration steps
%      d : size of system. Default = 22
%      np : saving spacing for orbit. Default = 1
%      nqr: saving spacing for Jacobian. Default = 1
% output:
%      aa : the orbit.  size [30, nstp/np+1]
% exmple usage:
%   aa = ksint(a0, 0.25, 1000);

if nargin < 6, d = 22; end
if nargin < 5, nqr = 1; end
if nargin < 4, np = 1; end

if nargout == 1,
    aa = MEXksint(a0, d, h, nstp, np, 1, 0, 0, 0, 0, 0); 
else if nargout ==2,
   [aa, daa] = MEXksint(a0, d, h, nstp, np, nqr, 0, 0, 0, 0, 1);
end

end
