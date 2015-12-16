function [ Model, acc ] = train_svm( X, Y, train_data, train_label )
% HYPER PARAMS: C, k(x,y).
    
    KERNELIZED = true;
    TRANI_PROPORTION = 1.0;
    ARTIFICIAL_PROP = 0.4;
    CLZ_NUM = 10;
    PCA_NUM = 200;
    
    options = optimoptions('quadprog', 'TolCon', 1e-8, ...
        'TolFun', 1e-8, 'MaxIter', 1000);
    
    Y = double(Y);

    [N, M] = size(X);
    
    % make more samples
    art_x = augment(X(1:N * ARTIFICIAL_PROP, :));
    art_y = Y(1:N * ARTIFICIAL_PROP, :);
    X = [X;art_x];
    Y = [Y;art_y];
    N = (1 + ARTIFICIAL_PROP) * N;
    
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
    
    hog = extract_hog(x_train, 'dala');
    
    %PCA
    [coeff,score] = pca_wairi(hog, PCA_NUM);
    hog = score;
    
%     hog = norm2_normalize(hog);
    
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
    
    min_c = 0.0022;
    max_c = 0.0022;
    step_sz = 0.0001;
    for c = min_c:step_sz:max_c
        ub = c * ones(train_size, 1);
        for  i = 1:CLZ_NUM
            fprintf('clz %d\n', i);
            a(:, i) = quadprog(H(:, :, i), f, [], [], [], [], lb, ub, [], options);
            b(i) = sum(a(:, i) .* y_train(:, i)) / sum(a(:, i) > 0);
        end
        m = struct('a', a, 'b', b, 'X', hog, 'Y', y_train, 'coeff', coeff);
        if TRANI_PROPORTION < 1.0
            % cross-validation
            y_pre = predict_svm(m, x_val);
            accu = sum(y_val == y_pre) / size(y_pre, 1);
            fprintf('c: %f, accuracy: %f\n', c, accu);
            if accu_max < accu
                Model = m;
            end
        else
            acc = 0;
            Model = m;
            y=classify(Model, train_data(0001:1000,:));acc = acc + sum(y==train_label(0001:1000,:))/1000;
            y=classify(Model, train_data(1001:2000,:));acc = acc + sum(y==train_label(1001:2000,:))/1000;
%             y=classify(Model, train_data(2001:3000,:));acc = acc + sum(y==train_label(2001:3000,:))/1000;
            y=classify(Model, train_data(3001:4000,:));acc = acc + sum(y==train_label(3001:4000,:))/1000;
            y=classify(Model, train_data(4001:5000,:));acc = acc + sum(y==train_label(4001:5000,:))/1000;
            fprintf('c: %f, acc: %f\n', c, acc / 4);
%             plot(c, acc / 4, '*');
            acc = acc / 4;
%             fprintf('pca: %d, acc: %f\n', np, acc);
        end
    end

    save('model_svm.mat', 'Model');
end