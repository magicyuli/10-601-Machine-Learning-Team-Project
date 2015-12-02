function [ normed ] = normlize( data )
%NORMLIZE Summary of this function goes here
%   Detailed explanation goes here
normed = bsxfun(@minus, data, mean(data, 1));
normed = bsxfun(@rdivide, normed, std(data, 1));
end

