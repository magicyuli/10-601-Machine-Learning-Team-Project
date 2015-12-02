function [ y ] = classify( model, X )
    y = predict_svm(model, X);
end

