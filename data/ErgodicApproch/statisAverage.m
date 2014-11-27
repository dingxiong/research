function average = statisAverage(dis, ang, Cell)

%Cell = 0.1;
dis = floor(dis/Cell);
N = length(dis);
minD = min(dis); maxD = max(dis); disp(minD); disp(maxD);
sumAng = zeros(maxD-minD+1, 1);
sumNum = zeros(maxD-minD+1, 1);
for i = 1:N,
    ix = dis(i) - minD +1;
    sumNum(ix) = sumNum(ix) + 1;
    sumAng(ix) = sumAng(ix) + ang(i); 
end

average = zeros(maxD-minD+1, 1); disp(sumNum); disp(sumAng);
for i = 1: length(sumNum),
   if sumNum(i) ~= 0, average(i) = sumAng(i)/sumNum(i); end
end

end