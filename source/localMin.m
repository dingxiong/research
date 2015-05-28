folder = {'./'}; pp = 'ppo4';
Mang = []; Mdis = [];
for k = 1:length(folder)
    num = dlmread([folder{k} 'No_' pp]); 
    ang = dlmread([folder{k} 'angle_' pp]);
    dis = dlmread([folder{k} 'dis_' pp]);

    N = length(num); s = 1;
    index = [];
    for i = 1:N
        range = s:s+num(i)-1;
        %[x, ix] = min(dis(range));
        [x, ix] = findpeaks(-dis(range));
        index = [index; s+ix];
        s = s + num(i);
    end

    Mang = [Mang; ang(index,:)];
    Mdis = [Mdis; dis(index)];

end