%function [t,X,Y,u] = cqCGL(A0, h, nstp)
% 1D cubic-quintic complex Gizburg-Landau equation on a periodic domain.
% A_t = Mu*A + (Br+1i*Bi)*A*abs(A)^2 + (Gr+1i*Gi)*A*abs(A)^4 + (Dr+1i*Di)*(u_xx + u_yy)
% Note Diffution constant is complex, then evaluation of ETDRK4
% coefficients should averaged on a whole contour circle.

Mu = -0.213; 
B = 1 + 0.8i; 
D = 0.125 + 0.5i;
G = -0.1 - 0.6i;
lam=50; N=256; % side length, grid points.


h = 0.01; nstp = 6000; np = 1; nqr = 50; isJ = 0;
A0 = gaussmf(linspace(0,lam,N), [2, lam/2])' * 0.7; v=fft(A0);
%v = uu(:,end);

k=[0:N/2-1 N/2 -N/2+1:-1]'*(2*pi)/lam; 
L = Mu - D * k.^2; 

% PRECOMPUTING ETDRK4 COEFFS 
E=exp(h*L); E2=exp(h*L/2);
M=64; % no. of points for complex mean
r=exp(2i*pi*(1:M)/M); % roots of unity
LR=h*L(:,ones(M,1))+r(ones(N,1),:);
Q = h * mean( (exp(LR/2)-1)./LR ,2);
f1 = h * mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2);
f2 = h * mean( (4+2*LR+exp(LR).*(-4+2*LR))./LR.^3 ,2);
f3 = h * mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2);

g1 = @(A)  B*A.*(abs(A).^2) + G*A.*abs(A).^4; 
g2 = @(A, dA)  B*(A.^2.*conj(dA) + 2*A.*conj(A).*dA) + ...
    G*(2*A.^3.*conj(A).*conj(dA) + 3*A.^2.*dA.*conj(A).^2); 
if isJ == 0, g = @(A) g1(A);
else g = @(A) [g1(A(:,1)), g2(repmat(A(:,1),1,2*N), A(:,2:end))];
end

if isJ == 1, 
    E = repmat(E, 1, 2*N+1); E2 = repmat(E2, 1, 2*N+1);
    f1 = repmat(f1, 1, 2*N+1); f2 = repmat(f2, 1, 2*N+1);
    f3 = repmat(f3, 1, 2*N+1); Q = repmat(Q, 1, 2*N+1); 
    
    v = [v, kron(eye(N),[1,1i])];
end

uu = zeros(N, floor(nstp/np)+1); uu(:,1) = v(:,1);
Qn = eye(2*N); lya = [];
for n = 1:nstp
    
    Nv = fft( g(ifft(v)) );     a = E2.*v + Q.*Nv; 
    Na = fft( g(ifft(a)) );     b = E2.*v + Q.*Na; 
    Nb = fft( g(ifft(b)) );     c = E2.*a + Q.*(2*Nb-Nv); 
    Nc = fft( g(ifft(c)) ); 
    
    v = E.*v + Nv.*f1 + (Na+Nb).*f2 + Nc.*f3; 
    
    if mod(n, np) == 0, uu(:,n/np+1) = v(:,1); end
    if isJ==1 && mod(n, nqr) ==0,
        J = zeros(2*N); J(1:2:end, :) = real(v(:,2:end)); 
        J(2:2:end, :) = imag(v(:,2:end)); 
        v(:,2:end) = kron(eye(N),[1,1i]);
        [Qn,Rn] = qr(J*Qn);  lya = [lya, log(abs(diag(Rn)))/(h*nqr)];
    end
end

%end
