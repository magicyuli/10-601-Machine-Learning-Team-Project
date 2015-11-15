function [ f, precision, recall ] = f_score(y, y_pre)
    precision = sum(y == y_pre & y == 1) / sum(y_pre == 1);
    recall = sum(y == y_pre & y == 1) / sum(y == 1);
    f = 2 / (1 / precision + 1 / recall);
end