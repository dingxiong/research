function [aa, daa] = ksfjaco(a0, h, nstp, d, np, nqr)
if nargin < 6, nqr = 1; end
if nargin < 5, np = 1; end
if nargin < 4, d = 22; end

if nargou == 2,
    [aa, daa] = MEXkssolve(a0, d, h, nstp, np, nqr, 1); 
elseif nargin == 1,
    aa = MEXkssolve(a0, d, h, nstp, np, 1, 1);
else
    fprinf("the number of output is wrong");
end

end