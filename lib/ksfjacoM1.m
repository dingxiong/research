function [tt, aa, daa] = ksfjacoM1(a0, h, nstp, d, np, nqr)
% integrate KS system without calculate Jacobian matrix 
% on the 1st mode slice.
% exmple usage:
%   aa = ksfjacoM1(a0, 0.25, 1000);

if nargin < 6, nqr = 1; end
if nargin < 5, np = 1; end
if nargin < 4, d = 22; end

if nargout == 2,
    [tt, aa] = MEXkssolve2(a0, d, h, nstp, np, 1, 0, 1); 
elseif nargout == 3,
    [tt, aa ,daa] = MEXkssolve2(a0, d, h, nstp, np, nqr, 1, 1);
else
    fprintf(1, 'the number of output is wrong. \n');
end


end
