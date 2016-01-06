function [ Model ] = train_NN( X, Y )
% Train a deep neural network

    % data augmentation
    [X, Y] = augment(X, Y);
    Y = double(Y);
    Y = bsxfun(@eq, Y, 0:9);

    % hog
    train_hog = extract_hog(X, 'dala');
    % pca
    [Evec, score] = pca_wairi(train_hog, 500); 
    train_hog = score;
    
    % input layer node number
    n_in = size(train_hog, 2);
    % output layer node number
    n_out = size(Y, 2);
    
    % number of hidden layers
    n_hid = 3;
    % total layers
    n_layers = n_hid + 2;
    % number of nodes in each layer
    n_nodes = [n_in 150 150 100 n_out];
    % NN configuration
    config = struct('n_layers', n_layers, 'n_nodes', n_nodes, ...
        'lambda', 0.001);
    % training
    M = NNs_converge(train_hog, Y, config);
    Model = struct('M', M, 'Evec', Evec);
end

