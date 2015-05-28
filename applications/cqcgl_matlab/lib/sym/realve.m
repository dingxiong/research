function rve = realve(ve)
% return the real space of those vectors
% input:
%   ve: eigenvectors

[n,m]=size(ve); rve=zeros(n,m);
i=1;
while i <= m
    if isreal(ve(:,i)), rve(:,i)=ve(:,i); i=i+1;
    else rve(:,i) = real(ve(:,i)); rve(:,i+1)=imag(ve(:,i)); i=i+2;
    end
end

end

