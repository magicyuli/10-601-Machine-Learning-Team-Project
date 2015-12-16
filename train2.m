function [ Model ] = train2( X, Y )
% Trigger point for training a kNN classifier
    
    Model = train_KNN(X, Y);
    Model2 = Model;
    save('Model2.mat', 'Model2');
end

