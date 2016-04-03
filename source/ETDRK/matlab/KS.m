global g;

N = 128;
x = 32*pi*(1:N)'/N;
u = cos(x/16).*(1+sin(x/16));
v = fft(u); v(2) = 1i*abs(v(2));

k = [0:N/2-1 0 -N/2+1:-1]'/16; % wave numbers
L = k.^2 - k.^4; % Fourier multipliers
g = -0.5i*k;

t0 = 0;
tend = 1550;
u0 = v;
h = 1/1; % time step
isReal = true;
skip_rate = 1;
rtol = 1e-6;
nu = 0.9;
mumax = 2.5;
mumin = 0.4;
mue = 1.25;
muc = 0.85;
doAdapt = true;

[tt, uu, duu, hs, NReject, NevaCoe] = ETDRK4B(L, @NL4, t0, u0, tend, h, skip_rate, isReal, ...
                                             doAdapt, rtol, nu, mumax, mumin, mue, muc);
Y = real(ifft(uu, N, 1));

%figure;
%plot(duu);

% Plot results:
%{
figure;
surf(tt, x, Y), shading interp, lighting phong, axis tight
view([-90 90]), colormap(autumn); set(gca,'zlim',[-5 50])
light('color',[1 1 0],'position',[-1,2,2])
material([0.30 0.60 0.60 40.00 1.00]);
%}