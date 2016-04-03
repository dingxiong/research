function zr = ZR(z, M, R, isReal)
% calulate the matrix zr for evaluating 
% function f(z)
% z        column vector
% M        samples on the circle
% R        redius of circle
% isReal   use only half plane or not

[M1, N] = size(z); 
assert(N == 1);

if isReal
    r = R*exp(1i*pi * ((1:M)-0.5) / M);
else
    r = R*exp(1i*2*pi * (1:M) / M);
end

zr = repmat(z, 1, M) + repmat(r, M1, 1);

end