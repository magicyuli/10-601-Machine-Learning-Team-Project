function [ Model ] = train1( X, Y )
% Trigger point for training a Neural Network classifier

    Model = train_NN(X, Y);
    Model1 = Model;
    save('Model1.mat', 'Model1');
end

