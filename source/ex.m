load ks22h02t100
%ang = sin(acos(dlmread('angle_ppo4')));
ang = dlmread('angle_ppo4');
dis = dlmread('dis_ppo4');
difv=dlmread('difv_ppo4');
No = dlmread('No_ppo4');
index = dlmread('indexPo_ppo4');

%x1 = 1; x2 = sum(No(1:1));
ix = 30;
x1=sum(No(1:ix))+1; x2=sum(No(1:ix+1));

a0=ppo(4).a; T=ppo(4).T; nstp=ppo(4).nstp;
aa1 = ksint(a0, T/nstp, 2*nstp);
raa1 = redSO2(aa1);

% original index starts
a1=raa1(:,index(x1)+1)+difv(x1,:)';
aa2 = ksint(a1, T/nstp, 2*nstp);
raa2 = redSO2(aa2);

raa3 = [];
for i =  x1:x2
    raa3 = [raa3, raa1(:, index(i)+1)+difv(i,:)'];
end

plot3(raa1(1,:), raa1(4,:), raa1(3,:), 'g'); hold on
%plot3(raa2(1,:), raa2(4,:), raa2(3,:), 'r');
plot3(raa3(1,:), raa3(4,:), raa3(3,:), 'b');
scatter3(raa3(1,1), raa3(4,1), raa3(3,1), 'm');

min = 230;
scatter3(raa3(1,min), raa3(4,min), raa3(3,min), 'm');

%%
thresh = 7e-3;
ixLargeDis = [];
for i = x1:x2
    if dis(i) > thresh, ixLargeDis = [ixLargeDis, i]; end
end
pre = 140;
subplot(221);
loglog(dis(x1:x2), ang(x1:x2,5), '.'); 
subplot(222);
loglog(dis(ixLargeDis), ang(ixLargeDis,5), '.');
subplot(223);
semilogy(dis(x1:x2))
subplot(224);
loglog(dis(x1+pre:x2), ang(x1+pre:x2,5), '.'); 

%% select those which has certain shadowing time.
threshT = 70/0.1;
N = length(No); 
s = 0; 
ixLargeT = []; 
for i = 1:N,
    if No(i)>threshT, ixLargeT = [ixLargeT, s+1:s+No(i)]; end
    s = s + No(i);
end

%% view certain instance
ang = dlmread('angle'); dis = dlmread('dis'); No = dlmread('No');
k1 = 28; k2 = 29;
start = sum(No(1:k1))+1; nd = sum(No(1:k2));
loglog(dis(start:nd), ang(start:nd, 5), '.');
