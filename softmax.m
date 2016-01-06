function [ o ] = softmax( z )
% softmax function
    o = exp(z);
    o = bsxfun(@rdivide, o, sum(o));
end