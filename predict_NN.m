function [indices] = predict_NN(X, Model)
test_hog = extract_hog(X);
% test_hog = normalize(double(X));
% test_hog = double(X);
% test_score = bsxfun(@minus, test_hog, mean(test_hog, 1)) * Model.Evec;

[~, indices] = max(sigmoid(bsxfun(@plus,Model.U * ...
    sigmoid(bsxfun(@plus,Model.W * test_hog', Model.B1)), Model.B2)));
indices = (indices - 1)';
%p = sum(indices == YTest) / size(YTest, 1);
end
