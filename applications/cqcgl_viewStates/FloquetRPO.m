% calculate Floquet exponents and vectors for po s in cqCGL

addpath('/usr/local/home/xiong/00git/research/lib/mex', '-end');
addpath('/usr/local/home/xiong/00git/KrylovSchur', '-end');

N = 1024;
d = 30;
fileName = '../../data/cgl/rpoT2X1.h5';
saveFiles = {'../../data/cgl/rpoT2X1_v2.h5', '../../data/cgl/rpoT2X1_v3.h5'};

for di = 0.4216:0.0001:0.4219
    
    fprintf(1, '==== di = %f ==== \n', di);
    
    [x, T, nstp, th, phi, err] = cqcglReadRPOdi(fileName, di, 1);
    h = T / nstp;
    Ndim = size(x, 1);

    gintgv = @(v) cqcglGintgv_threads(N, d, h, 1, 4.0, 0.8, 0.01, di, 4, x, v, ...
                                      th, phi, nstp);
    k = [15, 30];
    m = [25, 40];
    
    for i = 1:length(k)
        
        %[Q, H, isC, flag, nc, ni] = KrylovSchur(gintgv, rand(Ndim, 1), ...
        %                                        Ndim, k(i), m(i), ...
        %                                        50, 1e-16);
        [e, v, isC, flag, nc, ni] = KrylovSchurEig(gintgv, rand(Ndim, 1), ...
                                                   Ndim, k(i), m(i), 80, ...
                                                   1e-16);
        if flag == 1
            fprintf(1, 'k = %d not converged \n', k(i));
        end

        [~, id] = sort(abs(e), 'descend');
        e = e(id);
        v = v(:, id);
        
        vr = zeros(Ndim, size(v, 2));
        j = 1;
        while j <= size(v, 2)
            if isreal(v(:,j))
                vr(:, j) = v(:, j);
                j = j + 1;
            else
                vr(:, j) = real(v(:, j));
                vr(:, j+1) = imag(v(:, j));
                j = j + 2;
            end
        end

        ep = log(abs(e)) / T;
        w = angle(e);
    
        cqcglSaveRPOEVdi(saveFiles{i}, di, 1, [ep, w], vr);
        
    end

end