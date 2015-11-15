function [ model ] = train_svm( X, Y )
% HYPER PARAMS: C, k(x,y).

    [N, M] = size(X);
    train_size = N * 0.75;
    
    % TODO multi-class
    Y(Y ~= 0) = -1;
    Y(Y == 0) = 1;
    
    % train set
    x_train = X(1:train_size,:);
    y_train = Y(1:train_size,:);
    % validation set
    x_val = X(train_size + 1:N,:);
    y_val = Y(train_size + 1:N,:);
    
    hog = extract_hog(x_train);
    
    
    %%%%%%%%%%%%%%%%% no kernel %%%%%%%%%%%%%%%%%
    C = 100;
    
    H = (y_train * y_train') .* (hog * hog');
    f = -1 * ones(train_size, 1);
    lb = zeros(train_size, 1);
    
    % best model based on f score
    f_max = 0;
    c_max = 1;
    f_plot = zeros(C, 1);
    p_plot = zeros(C, 1);
    r_plot = zeros(C, 1);
    c_plot = zeros(C, 1);
    for c = 1:C
        ub = c * ones(train_size, 1);
        [a, fval, exitflag] = quadprog(H, f, [], [], [], [], lb, ub);
        b = sum(a .* y_train) / sum(a > 0);
        m = struct('a', a, 'b', b, 'X', hog, 'Y', y_train);
        % cross-validation
        y_pre = predict_svm(m, x_val);
        [f_s, p, r] = f_score(y_val, y_pre);
        if f_s > f_max
            f_max = f_s;
            c_max = c;
            model = m;
        end
        f_plot(c) = f_s;
        p_plot(c) = p;
        r_plot(c) = r;
        c_plot(c) = c;
        fprintf('c: %d, pre: %f, rec: %f, f: %f', c, p, r, f_s);
    end
    fprintf('c max: %d, f_max: %f', c_max, f_max);
    plot(c_plot, f_plot, c_plot, p_plot, c_plot, r_plot);
    legend('c', 'f', 'p', 'r');
end