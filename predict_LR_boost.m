function [ idx ] = predict_LR_boost(X, Model)
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
idx = zeros(testSize, 1);
for i = 1 : number
    proba = prob(X, Model.M(:,:,i));
    [~, indices] = max(proba, [], 2);
    indices = indices - 1;
    disp(indices);
    idx = idx + Model.Alpha(i,:) * indices;
end
end

