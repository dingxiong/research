function [x,average] = statisAverage(dis, ang, Cell)

%Cell = 0.1;
dis = floor(log10(dis)/Cell);
%dis = floor(dis/Cell);
N = length(dis);
minD = min(dis); maxD = max(dis);
sumAng = zeros(maxD-minD+1, size(ang,2));
sumNum = zeros(maxD-minD+1, 1);
for i = 1:N,
    ix = dis(i) - minD +1;
    sumNum(ix) = sumNum(ix) + 1;
    sumAng(ix,:) = sumAng(ix,:) + ang(i,:); 
end

% calcuate the mean value
average = zeros(maxD-minD+1, size(ang,2));
for i = 1: length(sumNum),
   if sumNum(i) ~= 0, average(i,:) = sumAng(i,:)/sumNum(i); end
end

% form x coordinate
x = 10.^(([minD:maxD]' + 0.5)*Cell);

end