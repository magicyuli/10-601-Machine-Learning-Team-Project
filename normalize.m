function [ normed, mu, sigma ] = normalize( data )
% Normalize the given data by centering and standardizing

    % mean
    mu = mean(data, 1);
    % standard deviation
    sigma = std(data, 1);

    normed = bsxfun(@minus, data, mu);
    normed = bsxfun(@rdivide, normed, sigma);
end

