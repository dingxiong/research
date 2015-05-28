function [aa, daa] = cqcglint(a0, nstp, np, nqr, h, d, Mu, B, D, G)

if nargin < 10, G = -0.1 - 0.6i; end
if nargin < 9, D = 0.125 + 0.5i; end
if nargin < 8, B = 1 + 0.8i;     end
if nargin < 7, Mu = -0.1;      end
if nargin < 6, d = 50;           end
if nargin < 5, h = 0.01;         end % default time step.
if nargin < 4, nqr =1;           end % default saving space for Jacobian.
if nargin < 3, np = 1;           end % default saving space for orbit.

if nargout == 1, isJ = 0; 
elseif nargout == 2, isJ = 1;
else disp('number of argout is wrong !');
end

if mod(length(a0),2) ~= 0, disp('dimension a0 is wrong !'); end
N = length(a0)/2;

if isJ == 0, aa = MEXcqcgl1d(a0, N, d, h, nstp, np, nqr, Mu, real(B), imag(B), real(D), imag(D), real(G), imag(G), 0);
else [aa, daa] = MEXcqcgl1d(a0, N, d, h, nstp, np, nqr, Mu, real(B), imag(B), real(D), imag(D), real(G), imag(G), 1);
end


end