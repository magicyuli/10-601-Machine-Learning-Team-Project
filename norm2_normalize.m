function [ normed ] = norm2_normalize( X )
    normed = X;
    for i = 1:size(X, 1)
        t = normed(i, :);
        normed(i, :) = t / norm(t, 2);
    end
end