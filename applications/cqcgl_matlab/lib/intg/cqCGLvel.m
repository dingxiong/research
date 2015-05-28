function velo = cqCGLvel(a, d, Mu, B, D, G)

if nargin < 6, G = -0.1 - 0.6i;  end
if nargin < 5, D = 0.125 + 0.5i; end
if nargin < 4, B = 1 + 0.8i;     end
if nargin < 3, Mu = -0.1;      end
if nargin < 2, d = 50;           end

a0 = RC(a, 'r2c'); N = length(a0); 
g1 = @(A) Mu*A + B*A.*(abs(A).^2) + G*A.*abs(A).^4; 

k=[0:N/2-1 N/2 -N/2+1:-1]'*(2*pi)/d; L = -D * k.^2;
velo = L.*a0 + fft(g1(ifft(a0)));
velo = RC(velo, 'c2r');

end