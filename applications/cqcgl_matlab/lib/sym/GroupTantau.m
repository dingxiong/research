function taa = GroupTantau(aa)

ca = RC(aa, 'r2c');
[N, M] = size(ca); 
k=[0:N/2-1 N/2 -N/2+1:-1]'; Tangent =1i*k;
cta = repmat(Tangent, 1, M) .* ca;
taa = RC(cta, 'c2r'); 

end
