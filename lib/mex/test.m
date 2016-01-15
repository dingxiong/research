N = 1024;
d = 30;
h = 0.0002;

Ndim = cqcglNdim(N);
a0 = 0.1 * ones(Ndim, 1);
v0 = rand(Ndim, 1);
nstp = 1000;
av = cqcglIntgv(N, d, h, 1, 4.0, 0.8, 0.01, 0.39, 4, a0, v0, nstp);