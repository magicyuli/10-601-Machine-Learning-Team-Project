function [ Model ] = train( X, Y )
% Trigger point for training a SVM classifier

    Model = train_svm(X, Y);
    save('Model.mat', 'Model');
end

