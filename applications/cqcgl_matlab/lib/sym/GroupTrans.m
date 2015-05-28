function raa = GroupTrans(aa, th, phi)
    
ca = RC(aa, 'r2c');
[n,m] = size(ca); 
k = [0:n/2,-n/2+1:-1]'; Group = exp( 1i*( phi + th*k ) );
cra = repmat(Group, 1, m) .* ca;
raa = RC(cra, 'c2r'); 
    
end