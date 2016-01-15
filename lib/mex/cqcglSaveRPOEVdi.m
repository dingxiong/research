function [] = cqcglSaveRPOEVdi(fileName, di, index, e, v)
    groupName = sprintf('%.6f/%d', di, index);
    cqcglSaveRPOEV(fileName, groupName, e, v);
end