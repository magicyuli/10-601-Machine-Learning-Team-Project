function [ y ] = classify1( Model, X )
%Trigger point for NN
    y = predict_NN(Model, X);
end

