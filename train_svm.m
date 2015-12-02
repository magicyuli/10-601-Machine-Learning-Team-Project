function [ Model ] = train_svm( X, Y )
% HYPER PARAMS: C, k(x,y).

    KERNELIZED = true;
    TRANI_PROPORTION = 1.0;
    CLZ_NUM = 10;

    [N, M] = size(X);
    train_size = N * TRANI_PROPORTION;
    
    % only 1 and -1
    y_binary = repmat(Y, 1, CLZ_NUM);
    
    for i = 1:CLZ_NUM
        indexes = y_binary(:, i) == i - 1;
        y_binary(indexes, i) = 1;
        y_binary(indexes == 0, i) = -1;
    end
    
    % train set
    x_train = X(1:train_size,:);
    y_train = y_binary(1:train_size,:);
    if TRANI_PROPORTION < 1.0
        % validation set
        x_val = X(train_size + 1:N,:);
        y_val = Y(train_size + 1:N,:);
    end
    
    hog = extract_hog(x_train);
    
    %PCA
%     [coeff,score] = pca(hog, 'Algorithm', 'eig', 'NumComponents', 900);
    [coeff,score] = pca_wairi(hog, 900);
    hog = score;
    
    % compute the [dot(x_i, x_j) * y_i * y_j] matrix (n by n)
    H = zeros(train_size, train_size, CLZ_NUM);
    
    if KERNELIZED
        A = kernel(hog, hog);
    else
        A = hog * hog';
    end
    
    for  i = 1:CLZ_NUM
        H(:, :, i) = (y_train(:, i) * y_train(:, i)') .* A;
    end
    
    f = -1 * ones(train_size, 1);
    lb = zeros(train_size, 1);
    
    % best model based on f score
    accu_max = 0;
    
    % init alpha
    a = zeros(train_size, CLZ_NUM);
    % init biase
    b = zeros(1, CLZ_NUM);
    
    min_c = 100;
    max_c = 100;
    for c = min_c:max_c
        ub = c * ones(train_size, 1);
        for  i = 1:CLZ_NUM
            fprintf('clz %d\n', i);
            a(:, i) = quadprog(H(:, :, i), f, [], [], [], [], lb, ub);
            b(i) = sum(a(:, i) .* y_train(:, i)) / sum(a(:, i) > 0);
        end
        m = struct('a', a, 'b', b, 'X', hog, 'Y', y_train, 'coeff', coeff);
        if TRANI_PROPORTION < 1.0
            % cross-validation
            y_pre = predict_svm(m, x_val);
            accu = sum(y_val == y_pre) / size(y_pre, 1);
            fprintf('c: %d, accuracy: %f\n', c, accu);
            if accu_max < accu
                Model = m;
            end
        else
            Model = m;
        end
    end

    save('model_svm.mat', 'Model');
end