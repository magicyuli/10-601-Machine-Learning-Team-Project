function [ Y ] = predict_svm( model, X )
    % no kernel
    hog = extract_hog(X);
    p = model.X * hog';
    h = model.a .* model.Y;
    Y = (((p)' * (h) + model.b) > 0);
end

