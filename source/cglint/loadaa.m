N = 1024; M = 10000;
fp = fopen('aa.bin'); AA = fread(fp, [2*N,M], 'double'); fclose(fp);
Ar = AA(1:2:end, :); Ai = AA(2:2:end, :); Ama = abs(Ar + 1i*Ai);

F=[];
for i = 1:M
    plot(Ama(:,i));
    %F = [F; getframe];
end
%movie(F);