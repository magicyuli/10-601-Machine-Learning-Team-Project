function [ Model ] = train_svm( X, Y )
% Train a kernel SVM classifier
    
    % ----------- START CONFIGURATION ---------- %
    KERNELIZED = true;
    CLZ_NUM = 10;
    PCA_NUM = 200;
    C = 0.0022;
    QP_OPTS = optimoptions('quadprog', 'TolCon', 1e-8, ...
        'TolFun', 1e-8, 'MaxIter', 1000);
    % ----------- END CONFIGURATION ---------- %
    
    % ----------- START PREPROCESSING ---------- %
    % data augmentation
    [X, Y] = augment(X, Y);
    % training sample size
    N = size(X, 1);
    
    % make Y binary
    Y = double(Y);
    Y = repmat(Y, 1, CLZ_NUM);
    for i = 1:CLZ_NUM
        indexes = Y(:, i) == i - 1;
        Y(indexes, i) = 1;
        Y(indexes == 0, i) = -1;
    end
    
    % hog
    hog = extract_hog(X, 'dala');
    % pca
    [coeff,score] = pca_wairi(hog, PCA_NUM);
    hog = score;
    % ----------- END PREPROCESSING ---------- %
    
    % ----------- START TRAINING ---------- %
    % compute dot products between samples
    if KERNELIZED
        A = kernel(hog, hog);
    else
        A = hog * hog';
    end
    
    % compute the [dot(x_i, x_j) * y_i * y_j] matrix (n by n)
    H = zeros(N, N, CLZ_NUM);
    for  i = 1:CLZ_NUM
        H(:, :, i) = (Y(:, i) * Y(:, i)') .* A;
    end
        
    % init alphas
    a = zeros(N, CLZ_NUM);
    % init biases
    b = zeros(1, CLZ_NUM);
    
    % do quadratic programming
    f = -1 * ones(N, 1);
    lb = zeros(N, 1);
    ub = C * ones(N, 1);
    for  i = 1:CLZ_NUM
        fprintf('Started training class %d\n', i);
        a(:, i) = quadprog(H(:, :, i), f, [], [], [], [], lb, ub, [], QP_OPTS);
        b(i) = sum(a(:, i) .* Y(:, i)) / sum(a(:, i) > 0);
    end
    % ----------- END TRAINING ---------- %
    
    Model = struct('a', a, 'b', b, 'X', hog, 'Y', Y, 'coeff', coeff);
end