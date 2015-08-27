po = [];
po1 = rpo1; po2 = rpo2; po3 = rpo3;
for i = 1:834,
    [a, b] = min([po1(i).r, po2(i).r, po3(i).r]);
    if b == 1, po = [po, po1(i)]; end
    if b == 2, po = [po, po2(i)]; end
    if b == 3, po = [po, po3(i)]; end
end
     