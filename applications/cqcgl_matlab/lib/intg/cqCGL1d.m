function [uu,da] = cqCGL1d(a0, nstp, h, lam, np, nqr, Mu, B, D, G)
% 1D cubic-quintic complex Gizburg-Landau equation on a periodic domain.
% A_t = Mu*A + (Br+1i*Bi)*A*abs(A)^2 + (Gr+1i*Gi)*A*abs(A)^4 + (Dr+1i*Di)*(u_xx + u_yy)
% Note Diffution constant is complex, then evaluation of ETDRK4
% coefficients should averaged on a whole contour circle.

if nargin < 10, G = -0.1 - 0.6i; end
if nargin < 9, D = 0.125 + 0.5i; end
if nargin < 8, B = 1 + 0.8i;     end
if nargin < 7, Mu = -0.213;      end
if nargin < 6, nqr =1;           end % default saving space for Jacobian.
if nargin < 5, np = 1;           end % default saving space for orbit.
if nargin < 4, lam = 50;         end
if nargin < 3, h = 0.01;         end % default time step.

if nargout == 1, isJ = 0; 
elseif nargout == 2, isJ = 1;
else disp('number of argout is wrong !');
end

if mod(length(a0),2) ~= 0, disp('dimension a0 is wrong !'); end
N = length(a0)/2; v = RC(a0,'r2c');

k=[0:N/2-1 N/2 -N/2+1:-1]'*(2*pi)/lam; 
L = Mu -D * k.^2; 

% PRECOMPUTING ETDRK4 COEFFS 
E=exp(h*L); E2=exp(h*L/2);
M=64; % no. of points for complex mean
r=exp(2i*pi*(1:M)/M); % roots of unity
LR=h*L(:,ones(M,1))+r(ones(N,1),:);
Q = h * mean( (exp(LR/2)-1)./LR ,2);
f1 = h * mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2);
f2 = h * mean( (4+2*LR+exp(LR).*(-4+2*LR))./LR.^3 ,2);
f3 = h * mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2);

g1 = @(A) B*A.*(abs(A).^2) + G*A.*abs(A).^4; 
g2 = @(A, dA) B*(A.^2.*conj(dA) + 2*A.*conj(A).*dA) + ...
    G*(2*A.^3.*conj(A).*conj(dA) + 3*A.^2.*dA.*conj(A).^2); 
if isJ == 0, g = @(A) g1(A);
else g = @(A) [g1(A(:,1)), g2(repmat(A(:,1),1,2*N), A(:,2:end))];
end

if isJ == 1, 
    E = repmat(E, 1, 2*N+1); E2 = repmat(E2, 1, 2*N+1);
    f1 = repmat(f1, 1, 2*N+1); f2 = repmat(f2, 1, 2*N+1);
    f3 = repmat(f3, 1, 2*N+1); Q = repmat(Q, 1, 2*N+1); 
    
    v = [v, kron(eye(N),[1,1i])];
    da = zeros(2*N*2*N, floor(nstp/nqr));
end

uu = zeros(2*N, floor(nstp/np)+1); uu(:,1) = a0;
for n = 1:nstp
    
    Nv = fft( g(ifft(v)) );     a = E2.*v + Q.*Nv; 
    Na = fft( g(ifft(a)) );     b = E2.*v + Q.*Na; 
    Nb = fft( g(ifft(b)) );     c = E2.*a + Q.*(2*Nb-Nv); 
    Nc = fft( g(ifft(c)) ); 
    
    v = E.*v + Nv.*f1 + (Na+Nb).*f2 + Nc.*f3; 
    
    if mod(n, np) == 0, uu(:,n/np+1) = RC(v(:,1)); end
    if isJ==1 && mod(n, nqr) ==0,
        J = zeros(2*N); J(1:2:end, :) = real(v(:,2:end)); 
        J(2:2:end, :) = imag(v(:,2:end)); 
        v(:,2:end) = kron(eye(N),[1,1i]);
        da(: , n/nqr) = J(:);
    end
end

end
