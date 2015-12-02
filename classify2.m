function [ y ] = classify2( model, X )
%CLASSIFY2 Summary of this function goes here
%   Detailed explanation goes here

y = predict_NN(X, model);

end

