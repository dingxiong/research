global c dr di;

d = 30;
N = 1024;
x = d*(1:N)'/N;
u = zeros(N, 1);
i1 = ceil(N/3); i2 = ceil(2*N/3);
u(i1:i2) = 3*rand(i2-i1+1, 1);
v = fft(u);
load v.mat; v(1) = 1i*abs(v(1));

b = 4.0;
c = 0.8;
dr = 0.01;
di = 0.06;

k = [0:N/2-1 N/2 -N/2+1:-1]' * 2*pi/d; % wave numbers
L = -(1+(1+1i*b)*k.^2); L(1) = 0;
%L = -((1-1i*176.67504941219335)+(1+1i*b)*k.^2); % Fourier multipliers


t0 = 0;
tend = 1.5;
u0 = v;
h = 1e-3; % time step
isReal = false;
skip_rate = 1;
rtol = 1e-6;
nu = 0.9;
mumax = 2.5;
mumin = 0.4;
mue = 1.25;
muc = 0.85;
doAdapt = true;

[tt, uu, duu, hs, NReject, NevaCoe] = ETDRK4B(L, @NL3, t0, u0, tend, h, skip_rate, isReal, ...
                                              doAdapt, rtol, nu, mumax, mumin, mue, muc);
Y = abs(ifft(uu, N, 1));
Y = Y';
%HeatMap(Y(:,1:10:end));

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