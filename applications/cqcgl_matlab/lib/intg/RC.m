function b = RC(a,di)
% split/combine the real and imaginary parts of input a

if nargin < 2, di = 'c2r'; end
if strcmpi(di,'c2r')
    [n,m] = size(a); b = zeros(2*n,m);
    b(1:2:end, :) = real(a); b(2:2:end,:) = imag(a);
elseif strcmpi(di, 'r2c')
    b = a(1:2:end,:) + 1i*a(2:2:end,:);
end