function [ normed, mu, sigma ] = normalize( data )
%NORMLIZE Summary of this function goes here
%   Detailed explanation goes here
mu = mean(data, 1);
sigma = std(data, 1);

normed = bsxfun(@minus, data, mu);
normed = bsxfun(@rdivide, normed, sigma);

end

