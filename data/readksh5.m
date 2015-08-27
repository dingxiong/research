pp = 'rpo';
file = './myN32/ks22h02t100E.h5';
start = 17; nd = 17;
po = [];

for i = start:nd,
    groupName = ['/' pp '/' num2str(i) '/'];
    tmpPO.a = hdf5read(file, [groupName 'a']);
    tmpPO.T = hdf5read(file, [groupName 'T']);
    tmpPO.nstp = hdf5read(file, [groupName 'nstp']);
    tmpPO.e = hdf5read(file, [groupName 'e']);
    tmpPO.r = hdf5read(file, [groupName 'r']);
    if strcmpi(pp, 'rpo') == 1,
        tmpPO.s = hdf5read(file, [groupName 's']);
    end
    po = [po, tmpPO];
end

for i = start:nd,
    e = po.e; T = po.e;
    N = size(e, 1);
    ep = zeros(N, 2);
    for j = 1:N,
        ep(j,:) = e(j,1:2);
        if e(j, 3) == 1,
            ep(j,2) = exp(1i*e(j,2));
        end
    end
    po.e = ep;
end

%%
N = 834;
po = rpo;
T = [];
T2= [];
T3= [];
for i = 1:N, T = [T; po(i).r]; end
for i = 1:N, 
    if T(i) > 1e-10, 
        T2 = [T2, i];
    else
        T3 = [T3, i];
    end, 
end

%% find marginal exponents
po = rpo;
marg = [];
for i = 1:200
    e = po(i).e(:,1);
    abse = sort(abs(e));
    marg =[marg; abse(1:2)];
end

%% transform the angle to phase
po = rpo;
newpo = [];
for i = 1:200
    pp = po(i);
    e = pp.e;
    for j = 1:62
        if(e(j,3)) == 1,
            e(j, 2) = exp(1i*e(j,2));
        end
    end
    pp.e = e(:, 1:2);
    newpo = [newpo, pp];
end