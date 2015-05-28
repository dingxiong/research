maxerr=[];
idx = [];
for i = 3:62,
    dot = v1(1:i,1:min(i, 30))' * v2(1:i, 1:min(i, 30));
    u = abs(triu(dot, 2)); u = max(u(:));
    v = abs(tril(dot, -2)); v = max(v(:));
    idx = [idx, i];
    maxerr=[maxerr, max(u,v)];
end