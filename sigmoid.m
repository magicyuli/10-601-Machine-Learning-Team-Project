function [ y ] = sigmoid( z )
% Sigmoid funtion
    y = 1 ./ (1 + exp(-z));
end

