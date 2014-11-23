th=[]
for k = 1: 5010
    ve = reshape(ve0(:,k), n, n);
    
    [q1, ~] = qr(ve(:,1:ix1), 0); [q2, ~] = qr(ve(:,ix1+1:end), 0);
    s = svd(q1'*q2); th1 = max(s); 
    
    [q1, ~] = qr(ve(:,1:ix2), 0); [q2, ~] = qr(ve(:,ix2+1:end), 0);
    s = svd(q1'*q2); th2 = max(s); 
    
    [q1, ~] = qr(ve(:,1:ix3), 0); [q2, ~] = qr(ve(:,ix3+1:end), 0);
    s = svd(q1'*q2); th3 = max(s); 
    
    th = [th; [th1, th2, th3]];
end

th =[]
for k = 1:5010
    ve = reshape(ve0(:,k), n, n);
    [q1, ~] = qr(ve(:,4), 0); [q2, ~] = qr(ve(:,5), 0);
    s = svd(q1'*q2); th1 = max(s);
    th = [th; th1];
end