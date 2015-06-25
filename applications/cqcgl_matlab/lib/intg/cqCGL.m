function [t,X,Y,u] = cqCGL(A0, h, nstp)
% 2D cubic-quintic complex Gizburg-Landau equation on a periodic domain.
% A_t = Mu*A + (Br+1i*Bi)*A*abs(A)^2 + (Gr+1i*Gi)*A*abs(A)^4 + (Dr+1i*Di)*(u_xx + u_yy)
% Note Diffution constant is complex, then evaluation of ETDRK4
% coefficients should averaged on a whole contour circle.

Br = 1; Gr = -0.1; Dr = 0.125; Gi = -0.6; Di = 0.5;
Mu = -0.4; Bi = 1.0;

lam=100; N=512; % side length, grid points.
g=@(A) Mu*A + (Br+1i*Bi)*A.*(abs(A).^2) + (Gr+1i*Gi)*A.*abs(A).^4; %nonlinear function

x=(lam/N)*(1:N)'; [X,Y]=ndgrid(x,x); %{[X,Y,Z]=ndgrid(x,x,x)}
v=fftn(A0); %random IC. {randn(N,N,N)}
k=[0:N/2-1 0 -N/2+1:-1]'/(lam/(2*pi)); %wave numbers
[xi,eta]=ndgrid(k,k); %2D wave numbers. {[xi,eta,zeta]=ndgrid(k,k,k)}
L = -(Dr+1i*Di) * (eta.^2+xi.^2); %2D Laplacian. {-D*(eta.^2+xi.^2+zeta.^2)}

Fr= false(N,1); %High frequencies for de-aliasing
Fr([N/2+1-round(N/6) : N/2+round(N/6)])=1;
[alxi,aleta]=ndgrid(Fr,Fr); %{[alxi,aleta,alzeta]=ndgrid(Fr,Fr,Fr)}
ind=alxi | aleta; %de-aliasing index. {alxi | aleta | alzeta}

%=============== PRECOMPUTING ETDRK4 COEFFS =====================
E=exp(h*L); E2=exp(h*L/2);
M=64; % no. of points for complex mean
r=exp(2i*pi*(1:M)/M); % roots of unity
L=L(:); LR=h*L(:,ones(M,1))+r(ones(N^2,1),:); %{r(ones(N^3,1),:)}
Q = h * mean( (exp(LR/2)-1)./LR ,2);
f1 = h * mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2);
f2 = h * mean( (4+2*LR+exp(LR).*(-4+2*LR))./LR.^3 ,2);
f3 = h * mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2);
f1 = reshape(f1,N,N); f2 = reshape(f2,N,N); f3 = reshape(f3,N,N);
L = reshape(L,N,N); Q = reshape(Q,N,N); %{reshape(*,N,N,N)}

%==================== TIME STEPPING LOOP =======================
for n = 1:nstp
    t = n*h; 
    Nv = fftn( g(ifftn(v)) );     a = E2.*v + Q.*Nv; 
    Na = fftn( g(ifftn(a)) );     b = E2.*v + Q.*Na; 
    Nb = fftn( g(ifftn(b)) );     c = E2.*a + Q.*(2*Nb-Nv); 
    Nc = fftn( g(ifftn(c)) ); 
    
    v = E.*v + Nv.*f1 + (Na+Nb).*f2 + Nc.*f3; %update
    v(ind) = 0; % High frequency removal --- de-aliasing
end

u=(ifftn(v));

%end
