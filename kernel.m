function [ k ] = kernel(X1, X2)
    SIGMA = 1.9;

    N = size(X1, 1);
    M = size(X2, 1);
    k = zeros(N, M);
    for i = 1:M
       d = bsxfun(@minus, X1, X2(i, :));
       for j = 1:N
           t = d(j, :);
           k(j, i) = t * t';
       end
    end
    k = exp(k / (-2 * SIGMA^2));
end