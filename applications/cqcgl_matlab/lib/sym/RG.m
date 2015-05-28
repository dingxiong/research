function R=RG(theta,N, flag, RorI)
% RG() is only used for cqCGLe.

if strcmpi(flag, 'p')
    if ischar(theta)
        if strcmpi(RorI, 'i'), 
        R =
    else
        
    end
elseif strcmpi(flag, 's')
end

if ischar(theta)
    if strcmpi(theta,'r'), 
        R=ones(N,1); R(1:2:end)=-1; R=diag(R); % reflection matrix
    elseif strcmpi(theta, 'raM1'), % reflection after 1st mode reduction. 
        S=ones(N,1); S(1:4:end) = -1; S(2:4:end) = -1;
        R = ones(N,1); R(1:2:end) = -1;
        %  return a (N-1, N-1) dimensional matrix.
        R = S.*R; R=[R(1); R(3:end)]; R = diag(R); 
    elseif strcmpi(theta,'t'),
        R=kron(diag(1:N/2),[0,-1; 1,0]);   % group tangent
    end
else % group transform for a finite angle.
    R=zeros(N);
    for j=1:N/2
        R(2*j-1:2*j, 2*j-1:2*j) = [cos(j*theta), -sin(j*theta);... 
                                   sin(j*theta), cos(j*theta)];          
    end
end

end