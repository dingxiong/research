t=clock();
for i in r_[:100]:
    aa, daa = ksfjaco(a0, 0.25, 10000, isJ = 1);

clock() - t
