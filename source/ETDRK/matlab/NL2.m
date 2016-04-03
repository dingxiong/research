function Nu = NL2(~, u)

global c dr di;

A = ifft(u);
A2 =  real(A .* conj(A));
A3 = A .* A2;
A5 = A2.*A2.*A;

Nu = fft((1+1i*c)*A3 - (dr+1i*di)*A5);

end