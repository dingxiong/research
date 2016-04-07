function Nu = NL3(~, u)
% slice function
    
    Nu = NL2(0, u);
    
    tx = 1i * u;
    
    c0 = imag(u(1));
    
    Nu = Nu + real(Nu(1)) / c0 * tx;
    
    Nu(1) = 1i*imag(Nu(1));

