function [ o ] = softmax( z )
% z: k by 1, where k is the hidden layer size
    o = exp(z);
    o = bsxfun(@rdivide, o, sum(o));
end