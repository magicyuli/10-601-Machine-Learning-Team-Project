function [ Y ] = KNN( Model, X_test )%, labels )
%KNN Summary of this function goes here
%   Detailed explanation goes here
%     figure;
%     hold on;
    % load data
    X_train = Model.X_train;
    Y_train = Model.Y_train;
    
%     K = 233;
    N = size(X_test, 1);
    Y_train = Y_train + 1;
    CLZ_NUM = max(Y_train);
    vote = zeros(N, CLZ_NUM);
    
%     acc_bst = 0;
    
    X_test = extract_hog(X_test, 'std');
    [Evec, X_train] = pca_wairi(extract_hog(X_train, 'std'), 143);
    X_test = bsxfun(@minus, X_test, mean(X_test, 1)) * Evec;
    
    [X_train, mu, sigma] = normalize(X_train);
    X_test = bsxfun(@minus, X_test, mu);
    X_test = bsxfun(@rdivide, X_test, sigma);
    
    sim = X_test * X_train';
    [sorted_all, sorted_idx_all] = sort(sim, 2, 'descend');
    for K = 233:233
        sorted = sorted_all(:, 1:K);
        sorted_idx = sorted_idx_all(:, 1:K);
        for i = 1:N
            v = zeros(1, CLZ_NUM);
            tmp_sim = sorted(i, :);
            tmp_idx = sorted_idx(i, :);
            for j = 1:K
                clz = Y_train(tmp_idx(j));
                v(clz) = v(clz) + tmp_sim(j);
            end
            vote(i, :) = v;
        end
        [~, Y] = max(vote, [], 2);
        Y = Y - 1;
%         accu = sum(Y==labels)/10000;
%         fprintf('K: %d, accu: %f\n', K, accu);
%         plot(K, accu, '*');
%         if acc_bst < accu
%             acc_bst = accu;
%             k_bst = K;
%         end
    end
%     fprintf('best K: %d, best accu: %f\n', k_bst, acc_bst);
end
