function taa = GroupTanrho(aa)

ca = RC(aa, 'r2c');
cta = 1i* ca;
taa = RC(cta, 'c2r'); 

end
