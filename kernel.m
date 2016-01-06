function [ k ] = kernel(X1, X2)
% Kernel function for SVM
%     TYPE = 'GAUSSIAN';
    TYPE = 'POLY';
%     TYPE = 'LAPLAC';

%     SIGMA_GAUSSIAN = 1.9;
%     SIGMA_LAPLAC = 22;

    % X1 sample count
    N = size(X1, 1);
    % X2 sample count
    M = size(X2, 1);
    
    % kernel matrix
    k = zeros(N, M);
    
    switch TYPE
        % Gaussian kernel
        case 'GAUSSIAN'
            for i = 1:M
               d = bsxfun(@minus, X1, X2(i, :));
               for j = 1:N
                   t = d(j, :);
                   k(j, i) = t * t';
               end
            end
            k = exp(k / (-2 * SIGMA_GAUSSIAN^2));
        % Laplacian kernel
        case 'LAPLAC'
            for i = 1:M
               d = bsxfun(@minus, X1, X2(i, :));
               for j = 1:N
                   k(j, i) = norm(d(j, :), 1);
               end
            end
            k = exp(k / -SIGMA_LAPLAC);
        % Polynomial kernel
        case 'POLY'
            k = X1 * X2';
            k = k.^3;
    end
end