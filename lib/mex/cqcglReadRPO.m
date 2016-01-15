function [x, T, nstp, th, phi, err] = cqcglReadRPO(fileName, ...
                                                   groupName)
    rpo = ['/' groupName '/'];
    x = h5read(fileName, [rpo, 'x']);
    T = h5read(fileName, [rpo, 'T']);
    nstp = double(h5read(fileName, [rpo, 'nstp']));
    th = h5read(fileName, [rpo, 'th']);
    phi = h5read(fileName, [rpo, 'phi']);
    err = h5read(fileName, [rpo, 'err']);
    
end