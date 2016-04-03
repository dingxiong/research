function [E, E2, Q, b1, b2, b4] = ETDRK4coe(L, h, isReal)

hL = h*L;
LR = ZR(hL, 32, 1, isReal);

E = exp(hL); 
E2 = exp(hL/2);

Q =  h * mean( (exp(LR/2)-1)./LR , 2);
b1 = h * mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 , 2);
b2 = h * 2 * mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 , 2);
b4 = h * mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 , 2);

if isReal
    Q = real(Q);
    b1 = real(b1);
    b2 = real(b2);
    b4 = real(b4);
end

end