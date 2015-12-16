function [Model] = train_NN( X, Y, val_data, val_label )
%TRAIN_NN Summary of this function goes here
%   Detailed explanation goes here
    Y = bsxfun(@eq, Y, 0:9);

    train_hog = extract_hog(X, 'dala');
    % train_hog = normalize(double(X));

    [Evec, score] = pca_wairi(train_hog, 500); train_hog = score;
%     Evec = 0;
    
    n_in = size(train_hog, 2);
    n_out = size(Y, 2);
    
    n_hid = 3;
    n_layers = n_hid + 2;
    n_nodes = [n_in 150 150 100 n_out];
    config = struct('n_layers', n_layers, 'n_nodes', n_nodes, ...
        'lambda', 0.001);

    M = NNs_converge(train_hog, Y, config, val_data, val_label, Evec);
    Model = struct('M', M, 'Evec', Evec);
    save('Model4.mat', 'Model');
end

