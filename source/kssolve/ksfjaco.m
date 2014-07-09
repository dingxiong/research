function [aa, daa] = ksfjaco(a0, h, nstp, d, np, nqr)
% integrate KS system with/without calculate Jacobian matrix.
% exmple usage:
  %   aa = ksfjaco(a0, 0.25, 1000);
% [aa, daa] = ksfjaco(a0, 0.25, 1000);

if nargin < 6, nqr = 1; end
if nargin < 5, np = 1; end
if nargin < 4, d = 22; end

if nargout == 2,
    [aa, daa] = MEXkssolve(a0, d, h, nstp, np, nqr, 1); 
elseif nargout == 1,
    aa = MEXkssolve(a0, d, h, nstp, np, 1, 1);
else
    fprintf(1, 'the number of output is wrong. \n');
end

end
