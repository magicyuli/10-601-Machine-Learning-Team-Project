function [ k ] = kernel(X1, X2)
%     TYPE = 'GAUSSIAN';
    TYPE = 'POLY';
%     TYPE = 'LAPLAC';
%     SIGMA_GAUSSIAN = 0.6;
    SIGMA_GAUSSIAN = 25;
    SIGMA_LAPLAC = 22;
    RHO_CHI_SQR = 3;

    N = size(X1, 1);
    M = size(X2, 1);
    k = zeros(N, M);
    
    switch TYPE
        case 'GAUSSIAN'
            for i = 1:M
               d = bsxfun(@minus, X1, X2(i, :));
               for j = 1:N
                   t = d(j, :);
                   k(j, i) = t * t';
               end
            end
            k = exp(k / (-2 * SIGMA_GAUSSIAN^2));
        case 'LAPLAC'
            for i = 1:M
               d = bsxfun(@minus, X1, X2(i, :));
               for j = 1:N
                   k(j, i) = norm(d(j, :), 1);
               end
            end
            k = exp(k / -SIGMA_LAPLAC);
        case 'POLY'
            k = X1 * X2';
            k = k.^3;
    end
end