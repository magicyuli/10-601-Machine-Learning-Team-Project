function [Model] = train_NN( X, Y )
%TRAIN_NN Summary of this function goes here
%   Detailed explanation goes here
labMat = bsxfun(@eq, Y, 0:9);

train_hog = extract_hog(X);

[Evec, score] = pca_wairi(train_hog, 700);
[U, W, B1, B2] = NNs_converge(score, labMat, 50, 0.01);
Model = struct('W', W, 'U', U, 'Evec', Evec, 'B1', B1, 'B2', B2);
save('Model2.mat', 'Model');
end

