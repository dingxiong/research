function [x, T, nstp, th, phi, err] = cqcglReadRPOdi(fileName, di, ...
                                                  index)
    groupName = sprintf('%.6f/%d', di, index);
    [x, T, nstp, th, phi, err] = cqcglReadRPO(fileName, groupName);

end