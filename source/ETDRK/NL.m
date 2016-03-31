function Nu = NL(~, u)

global g;

Nu = g .* fft(real(ifft(u)).^2);

end