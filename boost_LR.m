function [ Model ] = boost_LR( X, Y, number )
%BOOST_LR Summary of this function goes here
%Detailed explanation goes here
%hyperparameters
stop = 0.00001;
beta = 0.7;
lambda = 0.03;
sampleSize = size(X, 1);
batchSize = 900;


X = extract_hog(X);
[Evec, X] = pca_wairi(X, 200);
X = [X, ones(sampleSize, 1)];
labels = Y;
Y = bsxfun(@eq, Y, 0:9);
W = zeros(size(Y, 2), size(X, 2));
M = zeros(size(Y, 2), size(X, 2), number);
% store alpha in each iteration
Alpha = zeros(number, 1);
% helper variables
batchCount = (sampleSize + batchSize - 1) / batchSize;
converged = false;
n = 0;
epoch = 1;
last_log_l_h = -inf;

% initial sample weight
D = 1 / sampleSize * ones(sampleSize, 1);
fun = @(A, B) A .* B;
for learner = 1 : number
    % trian begin
    X = bsxfun(fun, X, D);
    while converged == false
        fprintf('=============start epoch %d=============\n', epoch);
        tmp_l_h = 0;
        for i = 1 : batchCount
            alpha = (100 + n) ^ -beta;
            start = (i - 1) * batchSize + 1;
            end1 = min(i * batchSize, sampleSize);
            % mini-batch x: k by m
            batch_x = X(start:end1, :);
            % mini-batch y: k by 5
            batch_y = Y(start:end1, :);
            % probability matrix A: k by 5
            A = prob(batch_x, W);
            A_log = log(A);
            tmp_l_h = tmp_l_h + sum(sum(batch_y .* A_log));
            % B: k by 5
            B = batch_y - A;
            % w: 5 by m
            W = W + alpha * (B.' * batch_x - lambda * W);
            n = n + 1;
            % disp((tmp_l_h - last_log_l_h) / -last_log_l_h);
        end
        % test the change in log likelihood
        if (tmp_l_h - last_log_l_h > 0) && ((tmp_l_h - last_log_l_h) / -last_log_l_h < stop)
            converged = true;
            fprintf('converged\n');
        end
        last_log_l_h = tmp_l_h;
        fprintf( 'Alpha: %f\n' ,alpha);
        fprintf('Log likelihood: %f\n', last_log_l_h);           
        fprintf('=============end epoch %d=============\n',  epoch);
        epoch = epoch + 1;
    end
    % train end
    % predict
    proba = prob(X, W);
    [~, indices] = max(proba, [], 2);
    indices = indices - 1;
    acc = sum(indices == labels) / sampleSize;
    % calculate error
    err = sum(0.5 * D * acc * sampleSize) / sum(D);
    alpha = log((1 - err) / err) / 2;
    Alpha(learner,:) = alpha;
    % update sample weight Dt+1(i)
    D = D .* exp( -alpha * (indices .* labels));
    D = D / norm(D);
    M(:, :, learner) = W;
    % Use a 3d array M, Model.M(:,:,i) = W
end

Model = struct('M', M, 'Evec', Evec, 'Alpha', Alpha);
save('Model101.mat', 'Model');
end
