function ra = Reflection(aa)
    
    ca = RC(aa, 'r2c');
    ra =[ca(1,:); flipud(ca(2:end,:))];
    ra = RC(ra, 'c2r');

end