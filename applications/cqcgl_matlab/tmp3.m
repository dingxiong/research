uu = cqCGL1d(RI(a,'i2r'), 20000, 0.005, 50, 10); x = ifft(RI(uu, 'r2i'));
for i = 1:size(x,2)  
    plot(abs(x(:,i))); axis([0, 256, -2.5, 2.5]); 
    getframe; 
end

for i = 1:size(x,2)  
    plot(real(x(:,i))); axis([0, 256, -2.5, 2.5]); 
    getframe; 
end

%% look at each power
N = size(uu,1);
for i = 1:N, plot(uu(i,1000:end)); title(num2str(i)); pause;  end
%% watch the solition explosion
xx = ifft(RC(aa, 'r2c')); xx = xx(:,1:2:end);
for i = 1:size(xx,2)  
    plot(abs(xx(:,i))); axis([0, 256, -4, 4]); 
    getframe; 
end
%% heat map
N = 256; d = 50; wid = 1000; 
A0 = gaussmf(linspace(0, d, N), [2, d/2])' * 0.7; a0=RC(fft(A0),'c2r');
aa = cqcglint(a0, wid, 2);
xx = ifft(RC(aa,'r2c')); xx=xx.';
imagesc(flipud(abs(xx))); colorbar; 

%% test different attractors
N = 256; d = 50; wid = 10000;
for i = 1 : 100
    A0 = 2*(rand(N,1)+1i*rand(N,1)); A0(1:floor(N/3)) = 0; A0(floor(2/3*N):end) = 0;
    a0 = RC(fft(A0));
    aa = cqcglint(a0, wid, 5);
    xx = ifft(RC(aa,'r2c')); 
    imagesc(flipud(abs(xx.'))); colorbar;
    pause;
end

%% test the C++ lyapunov code 
N = 256; L =50;
x = (1:N)/N*L -L/2;
a0 = exp(-x.^2/8); a0 = RC(fft(a0'),'c2r');
at = cqcglint(a0, 50000, 10000); a0 = at(:,end);
aa = cqcglint(a0, 20000, 20);
xx = ifft(RC(aa,'r2c'));
imagesc(flipud(abs(xx.'))); colorbar;

%% test different system size
N = 256; L = 50; Mu = -0.10; B = 1+0.8i; D = 0.125 + 0.5i;
x = (1:N)/N*L -L/2;
a0 = exp(-x.^2/8)*2; a0 = RC(fft(a0'),'c2r');
at = cqcglint(a0, 41000, 41000, 1, 0.01, L, Mu, B, D); a0 = at(:,end);
aa = cqcglint(a0, 700, 1, 1, 0.01, L, Mu, B, D); 
xx = ifft(RC(aa,'r2c'));
imagesc(flipud(abs(xx.'))); colorbar;

%% find the three different explosion structures
N = 256; L =50;
x = (1:N)/N*L -L/2;
a0 = exp(-x.^2/8); a0 = RC(fft(a0'),'c2r');
at = cqcglint(a0, 40000, 21000); a0 = at(:,end);
aa = cqcglint(a0, 14300, 1);
xx = ifft(RC(aa,'r2c'));
imagesc(flipud(abs(xx.'))); colorbar;

%% find the recurrence
[n,m]=size(aa); 
xx = RC(aa, 'r2c'); n = n/2;
th = zeros(m, 1); phi = zeros(m, 1);
for i = 1:m
    th1 = angle(xx(1,i)); th2 = angle(xx(2,i));
    ang1 = th2 - th1; ang2 = th1;
    xx(:,i) = exp(-1i*ang2)*exp(-1i*[0:n/2, -n/2+1:-1]'*ang1).*xx(:,i);
    th(i) = ang1; phi(i) = ang2;
end
%
for i=1:m
    %disp(i);
    for j = i+800 : m
        err = norm(xx(:,i)-xx(:,j));
        if(err < 9), disp([i,j,err]); end
    end
end
%%
x1 = xx(:, 1131); x2 = xx(:,4068);
th1 =angle(x1(1)); th2 = angle(x1(2)); th = th2 -th1; phi = 2*th1-th2;
th1 =angle(x2(1)); th2 = angle(x2(2)); th = th2 -th1; phi = 2*th1-th2;