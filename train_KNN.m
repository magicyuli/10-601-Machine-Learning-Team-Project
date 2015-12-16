function [ Model ] = train_KNN( X, Y )
% The training for kNN is just augmenting the data

    [X, Y] = augment(X, Y);
    Model = struct('X_train', X, 'Y_train', Y);
end