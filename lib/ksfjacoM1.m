function [tt, aa] = ksfjacoM1(a0, h, nstp, d, np, nqr)
% integrate KS system without calculate Jacobian matrix 
% on the 1st mode slice.
% exmple usage:
%   aa = ksfjacoM1(a0, 0.25, 1000);

if nargin < 6, nqr = 1; end
if nargin < 5, np = 1; end
if nargin < 4, d = 22; end

[tt, aa] = MEXkssolve(a0, d, h, nstp, np, 1, 0, 1); 

end
