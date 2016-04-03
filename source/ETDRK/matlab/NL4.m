function Nu = NL4(~, u)
    
N = length(u);

Nu = NL(0, u);

tx =  1i * [0:N/2-1 0 -N/2+1:-1]' .* u;

c1 = imag(u(2));

Nu = Nu + real(Nu(2)) / c1 * tx; 

Nu(2) = 1i*imag(Nu(2));

