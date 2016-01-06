function [ Y ] = predict_svm( Model, X )
    KERNELIZED = true;
    
    % number of test samples
    K = size(X, 1);
    
    % hog
    hog = extract_hog(X, 'dala');
    % pca
    test_score = bsxfun(@minus, hog, mean(hog, 1)) * Model.coeff;
    hog = test_score;
    
    % sum(dot(x_i, x_new)) (n by k)
    if KERNELIZED
        A = kernel(Model.X, hog);
    else
        A = Model.X * hog';
    end

    % [a_1 * y_1, a_2 * y_2, ...]' (n by CLZ_NUM)
    H = Model.a .* Model.Y;
    % Y = sigma(a_i * y_i * dot(x_i, x_new)) + b (k by CLZ_NUM)
    [~, Y] = max((A)' * (H) + repmat(Model.b, K, 1), [], 2);
    Y = Y - 1;
end

