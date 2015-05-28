function [raa, th, phi] = RedSym(aa)
% reduce the symmetries by the 0th and 1st modes

ca = RC(aa, 'r2c');
[n,m] = size(ca); cra = zeros(n,m);
phi = angle(ca(1,:)); th = angle(ca(2,:)) - phi;
k = [0:n/2,-n/2+1:-1]';
for i = 1:m
    cra(:,i) = exp( -1i*( phi(i) + th(i)*k ) ).*ca(:,i);
end

raa = RC(cra, 'c2r'); 

end