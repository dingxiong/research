function Dv = stab(a, d, Mu, B, D, G)
% calculate the stability matrix.
% Input a0 has size [N x 1], output Dv has size [2N x 2N].

if nargin < 6, G = -0.1 - 0.6i;  end
if nargin < 5, D = 0.125 + 0.5i; end
if nargin < 4, B = 1 + 0.8i;     end
if nargin < 3, Mu = -0.1;      end
if nargin < 2, d = 50;           end


a0 = RC(a,'r2c'); N = length(a0); 
da = kron(eye(N),[1,1i]);
A = ifft(a0); dA = ifft(da);
g2 = @(A, dA) Mu*dA + B*(A.^2.*conj(dA) + 2*A.*conj(A).*dA) + ...
    G*(2*A.^3.*conj(A).*conj(dA) + 3*A.^2.*dA.*conj(A).^2); 
k=[0:N/2-1 N/2 -N/2+1:-1]'*(2*pi)/d; L = -D * k.^2; L =repmat(L,1,2*N); 
dv = L.*da+fft(g2(repmat(A, 1, 2*N), dA));
Dv = zeros(2*N,2*N); Dv(1:2:end,:)=real(dv); Dv(2:2:end,:)=imag(dv);

end 