function [indices, p] = predict_NN(XTest, YTest, U, W)
[Y, indices] = max(f(U * f(W * XTest')));
indices = (indices - 1)';
p = sum(indices == YTest) / size(YTest, 1);
end

function x = f(x)
x = 1./(1+exp(-x));
end