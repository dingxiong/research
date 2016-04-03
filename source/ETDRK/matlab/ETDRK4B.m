function [tt, uu, duu, hs, NReject, NevaCoe] = ETDRK4B(L, NL, t0, u0, tend, h, skip_rate, ...
                                                      isReal, doAdapt, rtol, nu, mumax, ...
                                                      mumin, mue, muc)

[E, E2, a21, a31, a32, a41, a43, b1, b2, b4] = ETDRK4Bcoe(L, h, isReal);

Nt = ceil((tend-t0)/h);
N = length(u0);
M = floor(Nt / skip_rate) + 1;
uu = zeros(N, M);
tt = zeros(M, 1);
duu = zeros(M, 1);
hs = zeros(M, 1);

uu(:, 1) = u0;
tt(1) = t0;

num = 2;
t = t0;
u = u0;
NReject = 0;
NevaCoe = 1;

i = 0;
while t < tend
    i = i+1; if mod(i, 10000) == 0, disp(t); end
    
    if t + h > tend
        h = tend - t;
        [E, E2, a21, a31, a32, a41, a43, b1, b2, b4] = ETDRK4Bcoe(L, h, isReal);
        NevaCoe = NevaCoe + 1;
    end
    
    [unext, du] =  oneStep(NL, E, E2, a21, a31, a32, a41, a43, b1, b2, b4, t, u, h);

    if doAdapt
        s = nu * (rtol/du)^(1/4); 
        [mu, doChangh, doAccept] = adapth(s, mumax, mumin, mue, ...
                                          muc);
        if doAccept
            t = t + h;
            u = unext;
            if mod(i, skip_rate) == 0
                uu(:, num) = u;
                tt(num) = t;
                duu(num) = du;
                hs(num) = h;
                num = num + 1;
            end            
            
        else
            NReject = NReject + 1;
        end
        
        if doChangh
            h = mu * h;
            [E, E2, a21, a31, a32, a41, a43, b1, b2, b4] = ETDRK4Bcoe(L, h, isReal);
            NevaCoe = NevaCoe + 1;
        end
    else
        t = t + h;
        u = unext;
        if mod(i, skip_rate) == 0
            uu(:, num) = u;
            tt(num) = t;
            duu(num) = du;
            hs(num) = h;
            num = num + 1;
        end 
        
    end

end

end

function [unext, du] = oneStep(NL, E, E2, a21, a31, a32, a41, a43, b1, b2, b4, t, u, h)
    N1 = NL(t, u);
    
    U2 = E2.*u + a21.*N1;
    N2 = NL(t+h/2, U2);
    
    U3 = E2.*u + a31.*N1 + a32.*N2;
    N3 = NL(t+h/2, U3);
    
    U4 = E.*u + a41.*N1 + a43.*N3;
    N4 = NL(t+h, U4);
    
    U5 = E.*u + b1.*N1 + b2.*(N2+N3) + b4.*N4;
    N5 = NL(t+h, U5);
    
    du = norm(b4.*(N5 - N4)) / norm(U5); 
    
    unext = U5;
end
