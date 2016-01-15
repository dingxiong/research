addpath('/usr/local/home/xiong/00git/research/lib/mex', '-end');
addpath('/usr/local/home/xiong/00git/KrylovSchur', '-end');

N = 1024;
d = 30;
fileName = '../../data/cgl/rpoT2X1.h5';
saveFiles = [];

for di = [0.32, 0.36, 0.362:0.002:0.42, 0.421, 0.422, 0.4225, ...
          0.4226]
    
    fprintf(1, '==== di = %f ==== \n', di);
    
    [x, T, nstp, th, phi, err] = cqcglReadRPOdi(fileName, di, 1);
    h = T / nstp;
    Ndim = size(x, 1);

    gintgv = @(v) cqcglGintgv_threads(N, d, h, 1, 4.0, 0.8, 0.01, di, 4, x, v, ...
                                      th, phi, nstp);
    av = gintgv(rand(Ndim, 1));

    k = [8, 15];
    m = [15 25];
    
    for i = 1:length(k)
        %[Q, H, isC, flag, nc, ni] = KrylovSchur(gintgv, rand(Ndim, 1), ...
        %                                        Ndim, k, m, 50, 1e-16);
        [e, v, isC, flag, nc, ni] = KrylovSchurEig(gintgv, rand(Ndim, 1), ...
                                                   Ndim, k, m, 100, ...
                                                   1e-16);
        if flag == 1
            fprintf(1, 'k = %d not converged \n', k);
        end

        [~, id] = sort(abs(e), 'descend');
        e = e(id);
        v = v(:, id);
        
        vr = zeros(Ndim, size(v, 2));
        for j = 1:size(v,2)
            if isreal(v(:,j))
                vr(:, j) = v(:, j);
            else
                vr(:, j) = real(v(:, j));
                vr(:, j+1) = imag(v(:, j));
                j = j+1;
            end
        end

        ep = log(abs(e)) / T;
        w = angle(e);
    
        cqcglSaveRPOEVdi(saveFiles(i), di, 1, [ep, w], vr);
        
    end

end