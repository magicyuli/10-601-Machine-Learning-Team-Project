function [ y ] = classify( model, X )
%Trigger point for SVM
    y = predict_svm(model, X);
end

