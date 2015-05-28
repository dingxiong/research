function [f,A]=reqNewton(a)

N = length(a)/2;
tx1 = S1(a,'t'); tx2 = S2(a,'t'); % group tangent

f = cqCGLvel(a) + c*q1*tx1 + w*tx2; % funtion needs to be minimized.
if nargout ==2
    A = zeros(2*N+2, 2*N+2);
    A(1:2*N,1:2*N) = stab(a) + c*q1*S1('m',N) + w*S2('m',N);
    A(1:2*N,2*N+1) = q1*tx1; A(1:2*N,2*N+2) = tx2;
    A(2*N+1, 1:2*N) = tx1'; A(2*N+2, 1:2*N) = tx2';
end

end