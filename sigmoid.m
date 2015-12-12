function [ y ] = sigmoid( z )
%SIGMOID Summary of this function goes here
%   Detailed explanation goes here
y = 1./(1+exp(-z));
end

