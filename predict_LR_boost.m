function [ indices ] = predict_LR_boost(X, Model)
%PREDICT_LR_BOOST Summary of this function goes here
%   Detailed explanation goes here

%Predict the classes of samples.
%:param X: data samples
%:return: the classifications

X = extract_hog(X);
X = bsxfun(@minus, X, mean(X, 1)) * Model.Evec;
testSize = size(X, 1);
X = [X, ones(testSize, 1)];
% number of classifiers
number = size(Model.M, 3);
p = zeros(testSize, size(Model.M, 1));
for i = 1 : number
    proba = prob(X, Model.M(:,:,i));
    p = p + Model.Alpha(i,:) * proba;
end
[~, indices] = max(p, [], 2);
    indices = indices - 1;
end

