function [ normed ] = normalize( data )
%NORMLIZE Summary of this function goes here
%   Detailed explanation goes here
normed = bsxfun(@rdivide, data, std(data, 1));
normed = bsxfun(@minus, normed, mean(normed, 1));

end

