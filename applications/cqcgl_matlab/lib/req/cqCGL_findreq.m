% Find the equilibria and relative equilibria of KS system.
% Newton Iteration
maxn = 200; maxit = 10; tol = 10^-14; pretol = 0.1;  N = 256; d = 50;
q1 = 2*pi/d; w = 0; c =0;

for i = 1:100
    %m = floor(N/3); a = zeros(N,1); a(m+1:m*2) = rand(m,1); a = fft(a);
    a = rand(N,1)*2; a = fft(a);
    c = 0; w = 0;
    %c = 2*rand(1)-1; w = rand(1)*2*pi; 
    f = cqCGLvel(a,d) + c*q1*sym1('i', 't', a) + w*sym2('i','t',a);
    if norm(f) < pretol, break; end
end

%A0 = gaussmf(linspace(0,d,N), [2, d/2])' *2.4; a = fft(A0);

%%
load tmp.mat, 
for i = 1:maxn
    fprintf('=====    i = %d   ======\n', i);
    tx1 = sym1('i', 't', a); tx2 = sym2('i', 't', a);
    f = cqCGLvel(a) + c*q1*tx1 + w*tx2;
    if norm(f) < tol; fprintf(1,'stop at norm(f)=%g\n', norm(f)); break; end
    % construction of derivative.
    A = zeros(2*N+2, 2*N+2);
    A(1:2*N, 1:2*N) = stab(a) + c*q1*sym1('m', N, 0) + w*sym2('m', N, 0);
    A(1:2*N, 2*N+1) = RI(q1*tx1); A(1:2*N, 2*N+2) = RI(tx2);
    A(2*N+1, 1:2*N) = RI(tx1); A(2*N+2, 1:2*N) = RI(tx2);
    % solve A * dx = f;
    dx = -A\[RI(f);0;0]; %[dx, flag]=gmres(-A, [RI(f);0;0], 400, 10^-6, 400); %
    if flag ~= 0, fprintf(1, 'gmres does not converge\n'); end
    
    lam = 1;
    for j = 1:maxit
        anew = a + lam*RI(dx(1:2*N),'r2i');
        cnew = c + lam*dx(2*N+1); wnew = w + lam*dx(2*N+2);
        fnew = cqCGLvel(anew) + cnew*q1*sym1('i', 't', anew) + wnew*sym2('i','t',anew);
        if norm(fnew) < norm(f)
            a = anew; c = cnew; w = wnew;
            break;            
        else
            lam = lam/2;
        end
    end
    if j == maxit; break; end
end

%%
lam = 1;
[f,A] = reqNewton(a);
for i = 1:maxn     
    H = A'*A; H = H + lam*diag(diag(H)); dh = A'*f;
    Di=diag(1./diag(H)); U=triu(H,1); M=H+U'*Di*U;
    [da, res]=ConjGradPre(-H,dh,M,zeros(size(dh,1),1),maxit,eps);% use zeros initial condition.
    if size(res,1)==maxit, fprintf(1,'PCG not converge, res(end)=%g\n', res(end)); end
    anew = a + da(1:2*N); cnew = c + da(2*N+1); wnew = w +da(2*N+2);
    fnew = reqNewton(anew);
    if norm(fnew) < norm(f)
        a = anew; c = cnew; w = wnew;
        if norm(fnew) < tol, fprintf(1,'stop at norm(f)=%g\n', norm(fnew)); break; end
        lam = lam/10;
        [f,A] = reqNewton(anew);        
    else
        lam = lam*10;
        if lam>10^10; break; end
    end

end