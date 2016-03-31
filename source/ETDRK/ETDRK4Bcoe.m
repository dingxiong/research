function [E, E2, a21, a31, a32, a41, a43, b1, b2, b4] = ETDRK4Bcoe(L, h, isReal)

hL = h*L;
LR = ZR(hL, 32, 1, isReal);

E = exp(hL); 
E2 = exp(hL/2);

a21 = h * mean( (exp(LR/2)-1)./LR , 2);
a31 = h * mean( (exp(LR/2).*(LR-4)+LR+4)./LR.^2 , 2);
a32 = h * 2 * mean( (2*exp(LR/2)-LR-2)./LR.^2 , 2);
a41 = h * mean( (exp(LR).*(LR-2)+LR+2)./LR.^2 , 2);
a43 = h * 2 * mean( (exp(LR)-LR-1)./LR.^2 , 2);
b1 = h * mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 , 2);
b2 = h * 2 * mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 , 2);
b4 = h * mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 , 2);

if isReal
    a21 = real(a21);
    a31 = real(a31);
    a32 = real(a32);
    a41 = real(a41);
    a43 = real(a43);
    b1 = real(b1);
    b2 = real(b2);
    b4 = real(b4);
end

end