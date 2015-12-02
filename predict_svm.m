function [ Y ] = predict_svm( model, X )
   KERNELIZED = true;
    % K: number of test samples
    K = size(X, 1);
    
    hog = extract_hog(X);
    
    %PCA
    test_score = bsxfun(@minus, hog, mean(hog, 1)) * model.coeff;
    hog = test_score;
    
    % p = sigma(dot(x_i, x_new)) (n by k)
    if KERNELIZED
        p = kernel(model.X, hog);
    else
        p = model.X * hog';
    end

    % h = [a_1 * y_1, a_2 * y_2, ...]' (n by CLZ_NUM)
    h = model.a .* model.Y;
    % Y = sigma(a_i * y_i * dot(x_i, x_new)) + b (k by CLZ_NUM)
    [m, Y] = max((p)' * (h) + repmat(model.b, K, 1), [], 2);
    Y = Y - 1;
end

