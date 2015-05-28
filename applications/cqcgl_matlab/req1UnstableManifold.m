h5file = '/usr/local/home/xiong/svn/DOGS/blog/code/data/req.h5';
a0 = h5read(h5file, '/req1/a0');
w1 = h5read(h5file, '/req1/w1');
w2 = h5read(h5file, '/req1/w2');
ve = h5read(h5file, '/req1/vecr') + 1i * h5read(h5file, '/req1/veci');
val = h5read(h5file, '/req1/valr') + 1i * h5read(h5file, '/req1/vali');

i1 = 5; i2 = 7; i3 =9;
    
T = 2*pi/imag(val(1)); nstp = round(T/0.01); h = T/nstp;
v1 = normc(real(ve(:,1))); v2 = normc(imag(ve(:,1)));
v4 = normc(real(ve(:,3))); v3 = normc(imag(ve(:,3)));
tol = 10^-5;  d0 = 10^-2;

nn = 1;
for i = 0:nn-1
    th = i/nn*2*pi;
    aa = cqcglint(a0 + d0*(cos(th)*v1 + sin(th)*v2), nstp*140, 1);
    [raa,~,~] = RedSym(aa);
    plot3(raa(i1,:), raa(i2,:), raa(i3,:), 'r'); hold on;
    
    aa = cqcglint(a0 + d0*(cos(th)*v3 + sin(th)*v4), nstp*140, 1);
    [raa,~,~] = RedSym(aa);
    plot3(raa(i1,:), raa(i2,:), raa(i3,:), 'b'); hold on
    
    %pause
end
% reflection point
[a0Red, ~,~]= RedSym(a0);
scatter3(a0Red(i1), a0Red(i2), a0Red(i3), 'k');
a0Ref = Reflection(a0); [a0RefRed,~,~] = RedSym(a0Ref);
scatter3(a0RefRed(i1), a0RefRed(i2), a0RefRed(i3), 'k');
%
aa = cqcglint(a0+rand(512,1)*0.01, 4000, 4000); x0 = aa(:,end); 
aa = cqcglint(x0, 600,1);  [raa,~,~] = RedSym(aa);
plot3(raa(i1,:), raa(i2,:), raa(i3,:), 'y');


%%
m = size(raa,2);
for i = round(m/1.7):1:m
    scatter3(raa(1,i), raa(3,i), raa(5,i), 'k'); hold on
    getframe
end

%% version 2
h5file = '/usr/local/home/xiong/svn/DOGS/blog/code/data/req.h5';
a0 = h5read(h5file, '/req1/a0');
w1 = h5read(h5file, '/req1/w1');
w2 = h5read(h5file, '/req1/w2');
ve = h5read(h5file, '/req1/vecr') + 1i * h5read(h5file, '/req1/veci');
val = h5read(h5file, '/req1/valr') + 1i * h5read(h5file, '/req1/vali');
    
T = 2*pi/imag(val(1)); nstp = round(T/0.01); h = T/nstp;
tol = 10^-5;  d0 = 10^-2;

v1 = normc(real(ve(:,1))); v2 = normc(imag(ve(:,1)));
v3 = normc(real(ve(:,3))); v4 = normc(imag(ve(:,3)));
v7 = normc(real(ve(:,7))); v8 = normc(imag(ve(:,7)));

[a0Red, th, phi] = RedSym(a0);
veProj = ProjVec(realve(ve), a0Red, th, phi); 
veProjNorm = normc(veProj);
Base = [veProjNorm(:,1), veProjNorm(:,9), veProjNorm(:,7)];
[q,r] = qr(Base,0); Base = q;

CoeTrans = [veProj(:,1), veProj(:,2), veProj(:,3)]' * Base;

nn = 1;
for i = 0:nn-1
    th = i/nn*2*pi;
    aa = cqcglint(a0 + d0*(cos(th)*v2 + sin(th)*v3), nstp*140, 1, 1, 0.01);
    [raa,~,~] = RedSym(aa);
    raa = raa - repmat(a0Red, 1, size(raa,2)); %raa = raa(:, end - 1500:end);
    ProjCoe = Base' * raa;
    plot3(ProjCoe(1,:), ProjCoe(2,:), ProjCoe(3,:), 'r'); hold on;
    scatter3(ProjCoe(1,end), ProjCoe(2,end), ProjCoe(3,end), 'k')
    
    %
    aa = cqcglint(a0 + d0*(cos(th)*v3 + sin(th)*v4), nstp*140, 1);
    [raa,~,~] = RedSym(aa);
    raa = raa - repmat(a0Red, 1, size(raa,2));
    ProjCoe = Base' * raa;
    plot3(ProjCoe(1,:), ProjCoe(2,:), ProjCoe(3,:), 'b'); hold on;
    %}
    %pause
end

% reflection point
a0Ref = Reflection(a0); [a0RefRed,~,~] = RedSym(a0Ref);
a0RefRed = a0RefRed - a0Red;
a0RefProjCoe = Base' * a0RefRed;
scatter3(a0RefProjCoe(1), a0RefProjCoe(2), a0RefProjCoe(3), 'k');
%
aa = cqcglint(a0+rand(512,1)*0.01, 4000, 4000); x0 = aa(:,end);
aa = cqcglint(x0, 600,1);  
[raa,~,~] = RedSym(aa); 
raa = raa - repmat(a0Red, 1, size(raa,2));
ProjCoe = Base' * raa;
plot3(ProjCoe(1,:), ProjCoe(2,:), ProjCoe(3,:), 'y'); hold on;
% test on the plane
[subsp,~] = qr(veProjNorm(:,1:4), 0 );
maxcos=[];
for i = 1:size(raa,2)
    sigma = subsp' * normc(raa(:,i));
    maxcos = [maxcos, max(sigma)];
end
%%
figure(2)
for i = 1: size(aa,2)
    scatter3(ProjCoe(1,i), ProjCoe(2,i), ProjCoe(3,i), 'r'); hold on;
    getframe;
end

%%
% use stability of travelling wave to observe the tree types of 
% explosion.
h5file = '/usr/local/home/xiong/svn/DOGS/blog/code/data/req.h5';
a0 = h5read(h5file, '/req1/a0');
w1 = h5read(h5file, '/req1/w1');
w2 = h5read(h5file, '/req1/w2');
ve = h5read(h5file, '/req1/vecr') + 1i * h5read(h5file, '/req1/veci');
val = h5read(h5file, '/req1/valr') + 1i * h5read(h5file, '/req1/vali');

a0Ref = Reflection(a0); 
T = 2*pi/imag(val(1)); nstp = round(T/0.01); h = T/nstp;
v1 = imag(ve(:,1))-real(ve(:,1)); 
aa = cqcglint(Reflection(a0) + Reflection(v1)*0.01, nstp*250, 1);

