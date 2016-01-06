function [ Y ] = KNN( Model, X_test )
%Classifier based on KNN 
%   When decide predictions, the K nearest neighbors vote using their
%   distances to the test samples

    X_train = Model.X_train;
    Y_train = Model.Y_train;
    Y_train = Y_train + 1;
    
    K = 233;
    PCA_NUM = 143;
    CLZ_NUM = max(Y_train);
    
    N = size(X_test, 1);
    vote = zeros(N, CLZ_NUM);

    % hog
    X_test = extract_hog(X_test, 'std');
    % pca
    [Evec, X_train] = pca_wairi(extract_hog(X_train, 'std'), PCA_NUM);
    X_test = bsxfun(@minus, X_test, mean(X_test, 1)) * Evec;
    
    % normalize
    [X_train, mu, sigma] = normalize(X_train);
    X_test = bsxfun(@minus, X_test, mu);
    X_test = bsxfun(@rdivide, X_test, sigma);
    
    % compute dot product similarities for every test point
    sim = X_test * X_train';
    % sort
    [sorted_all, sorted_idx_all] = sort(sim, 2, 'descend');
    % get k nearest neighbors for every test point
    sorted = sorted_all(:, 1:K);
    sorted_idx = sorted_idx_all(:, 1:K);
    % predict labels
    for i = 1:N
        % vote vector
        v = zeros(1, CLZ_NUM);
        tmp_sim = sorted(i, :);
        tmp_idx = sorted_idx(i, :);
        % add similarities to the vote vector
        for j = 1:K
            clz = Y_train(tmp_idx(j));
            v(clz) = v(clz) + tmp_sim(j);
        end
        vote(i, :) = v;
    end
    % get the labels which have largest votes
    [~, Y] = max(vote, [], 2);
    Y = Y - 1;
end
