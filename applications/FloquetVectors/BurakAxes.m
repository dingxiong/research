addpath('/usr/local/home/xiong/svn/siminos/ksConnected')
load('/usr/local/home/xiong/svn/siminos/matlab/ruslan/kse22orbits.mat');

epsilon = 1e-4;
%Function which should be satisfied by an equilibrium: 
feq = @(eq) vel(eq); %No constraint
%Function which should be satisfied by a relative equilibrium: 
ftw = @(tw) [velred(tw); SliceCond(tw)]; %Note that the constraint is the slice
                                         %condition. 
eq1 = eq(1).a; %Pick eq2, this is 62 dim vector (solved using 32 modes)
eq1 = eq1(1:30); %We use 16 modes (first mode is 0, hence real valued state 
                 %space vector is 30 dimensional                 

eq2 = eq(2).a; %Pick eq2, this is 62 dim vector (solved using 32 modes)
eq2 = eq2(1:30); %We use 16 modes (first mode is 0, hence real valued state 
                 %space vector is 30 dimensional

eq3 = eq(3).a; %Pick eq2, this is 62 dim vector (solved using 32 modes)
eq3 = eq3(1:30); %We use 16 modes (first mode is 0, hence real valued state 
                 %space vector is 30 dimensional                 
                 
tw1 = re(1).a; %Pick tw1, this is 62 dim vector (solved using 32 modes)
tw1 = tw1(1:30); %We use 16 modes (first mode is 0, hence real valued state 
                 %space vector is 30 dimensional
                 
tw2 = re(2).a; %Pick tw1, this is 62 dim vector (solved using 32 modes)
tw2 = tw2(1:30); %We use 16 modes (first mode is 0, hence real valued state 
                 %space vector is 30 dimensional                 
                 
tw1 = LieEl(-pi/2)*tw1; %This solution has Re[a_1] = x_1 = 0, we define the 
tw2 = LieEl(-pi/2)*tw2; %This solution has Re[a_1] = x_1 = 0, we define the 
eq1 = LieEl(-pi/2)*eq1; %This solution has Re[a_1] = x_1 = 0, we define the 
                        %slice as y_1 = 0, so we bring the solution onto it                                              

%Since the eq* and tw* from database came from a 32-mode solution, they 
%yield some numerical errors when we discard 16 modes. So we refine them:

[eq2, fval, info] = fsolve(feq, eq2);
[eq1, fval, info] = fsolve(feq, eq1); eq1(2) = abs(eq1(2));
[eq3, fval, info] = fsolve(feq, eq3);
[tw1, fval, info] = fsolve(ftw, tw1);                        
[tw2, fval, info] = fsolve(ftw, tw2);

%Unstable direction of eq2:
Aeq2 = gradV(eq2);
%Compute stability eigenvectors:
[V, lambda] = eig(Aeq2);
lambda=eig(Aeq2); %Stability eigenvalues
lambdarealsorted = sort(real(lambda), 'descend'); %Sort the real parts of evals
%Find the place of the most expanding eigenvalue on the list:
i1 = find(lambdarealsorted(1)==real(lambda)); 
i1 = i1(1); %Pick one of the complex pair
v1eq2 = real(V(:, i1)); v1eq2 = v1eq2/norm(v1eq2); %Real part of the most expanding one
%v1eq2 = imag(V(:, i1)); v1eq2 = v1eq2/norm(v1eq2); %Real part of the most expanding one

eq2Unstable = eq2 + epsilon * v1eq2;
eq2Unstablehat = x2xhat(eq2Unstable');
eq2tU = RefReduce4(eq2Unstablehat'); 


%Stable direction of eq2:
%Find the place of the least stable eigenvalue on the list:
i3 = find(lambdarealsorted(4)==real(lambda));
i3 = i3(1);
% The stable vector:
v3eq2 = real(V(:, i3)); v3eq2 = v3eq2/norm(v3eq2);

eq2Stable = eq2 + epsilon * v3eq2;
eq2Stablehat = x2xhat(eq2Stable');
eq2tS = RefReduce4(eq2Stablehat'); 

%Unstable direction of eq3:
Aeq3 = gradV(eq3);
%Compute stability eigenvectors:
[V, lambda] = eig(Aeq3);
lambda=eig(Aeq3); %Stability eigenvalues
lambdarealsorted = sort(real(lambda), 'descend'); %Sort the real parts of evals
%Find the place of the most unstable eigenvalue on the list:
i1 = find(lambdarealsorted(1)==real(lambda));
i1 = i1(1)
vUnstableEq3 = real(V(:, i1)); vUnstableEq3 = vUnstableEq3/norm(vUnstableEq3);
i2 = find(lambdarealsorted(2)==real(lambda));
i2 = i2(1)
vUnstable2Eq3 = real(V(:, i2)); vUnstable2Eq3 = vUnstable2Eq3/norm(vUnstable2Eq3);

%Stable direction of eq3:
%Find the place of the least stable eigenvalue on the list:

i5 = find(lambdarealsorted(5)==real(lambda));
i5 = i5(1);
% The stable vector:
vStableEq3 = real(V(:, i5)); vStableEq3 = vStableEq3/norm(vStableEq3);

i7 = find(lambdarealsorted(7)==real(lambda));
i7 = i7(1);
% The stable vector:
vStable2Eq3 = real(V(:, i7)); vStable2Eq3 = vStable2Eq3/norm(vStable2Eq3);


eq3Unstable = eq3 + epsilon * vUnstableEq3;
eq3Unstablehat = x2xhat(eq3Unstable');
eq3tU = RefReduce4(eq3Unstablehat'); 

eq3Unstable2 = eq3 + epsilon * vUnstable2Eq3;
eq3Unstablehat2 = x2xhat(eq3Unstable2');
eq3tU2 = RefReduce4(eq3Unstablehat2'); 

eq3Stable = eq3 + epsilon * vStableEq3;
eq3Stablehat = x2xhat(eq3Stable');
eq3tS = RefReduce4(eq3Stablehat'); 

eq3Stable2 = eq3 + epsilon * vStable2Eq3;
eq3Stablehat2 = x2xhat(eq3Stable2');
eq3tS2 = RefReduce4(eq3Stablehat2'); 

eq1t = RefReduce4(eq1); 
eq2t = RefReduce4(eq2); 
eq3t = RefReduce4(eq3); 
tw1t = RefReduce4(tw1); 
tw2t = RefReduce4(tw2); 

% Projection bases:

origin = eq2tS;
e1 = eq3tS - origin;
e2 = eq3tS2 - origin;
e3 = eq2tU - origin;
e4 = tw1t - origin;

% Gram-Schmidt:

e1 = e1 / norm(e1);
e2 = e2 - dot(e1,e2) * e1; e2 = e2 / norm(e2);
e3 = e3 - dot(e1, e3) * e1 - dot(e2, e3) * e2; e3 = e3 / norm(e3);
e4 = e4 - dot(e1, e4) * e1 - dot(e2, e4) * e2 - dot(e3, e4) * e3; e4 = e4 / norm(e4);
ee = [e1'; e2'; e3'; e1'; e2'; e4'];

eq1proj = ee * (eq1t - origin);
eq2sproj = ee * (eq2tS - origin);
eq2uproj = ee * (eq2tU - origin);
eq3s1proj = ee * (eq3tS - origin);
eq3u1proj = ee * (eq3tU - origin);
eq3s2proj = ee * (eq3tS2 - origin);
eq3u2proj = ee * (eq3tU2 - origin);
tw1proj = ee * (tw1t - origin);
tw2proj = ee * (tw2t - origin);