function G = S2(x, th)
% group transform; group tangent; group matrix for phase rotation
% symmetry of cqCGLe.
% Input:
%   th : char/th/N. If x is char, then th is used to indicate the dimension.
%   x : is the input vector / char to indicate that you want to get T. 

if ~ischar(x)
    N = length(x)/2; k = ones(N,1);
    G = zeros(2*N,1); 
    if ischar(th), 
       G(1:2:end) = -k.*x(2:2:end); G(2:2:end)=k.*x(1:2:end);
    else 
       cs = cos(k*th); sn = sin(k*th);
       G(1:2:end) = cs * x(1:2:end) - sn * x(2:2:end);
       G(2:2:end) = sn * x(1:2:end) + cs * x(2:2:end);
    end
else
    N = th; 
    k = ones(N,1);
    G = kron(diag(k), [0,-1;1,0]);
end

end