str = 'rpo3'; path = './rpo3/'; Col = 4; MaxN = 350;

ang = dlmread([path 'angle_' str]); ang = sin(acos(ang));
dis = dlmread([path 'dis_' str]); 
No = dlmread([path 'No_' str]);

N = min(length(No), MaxN); sum = 1;
Pos = [];
Sang = []; Sdis =[]; % really shadowing incidences.
scrsz = get(groot,'ScreenSize'); XL = scrsz(3); YL = scrsz(4);
for i = 31:N,
    disp(i);
    S = 'n';
    while strcmpi(S, 'n'),
        figure;
        plot(log10(dis(sum:sum+No(i)-1))); grid on;
        [x, ~] = ginput(4); x = round(x);
        close;
        
        figure('Position', [XL/3, YL/2, 1*XL/2, YL/3]);
        subplot(1,2,1)
        plot( log10(dis(sum+x(1):sum+x(2))), log10(ang(sum+x(1):sum+x(2), Col)), '.')
        axis equal
        subplot(1,2,2)
        plot( log10(dis(sum+x(3):sum+x(4))), log10(ang(sum+x(3):sum+x(4), Col)), '.')
        axis equal
        %axis([-3.2, -1, -8, -1]);
        S = input('Are you satisfied ? [Enter/n]', 's'); 
        close;
    end
    Pos = [Pos; x'];
    Sang = [Sang; ang(sum+x(1):sum+x(2),:); ang(sum+x(3):sum+x(4), :) ]; 
    Sdis = [Sdis; dis(sum+x(1):sum+x(2)); dis(sum+x(3):sum+x(4)) ];    
    sum = sum + No(i);
end

