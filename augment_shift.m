function [ aug ] = augment_shift( X )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    VEC_SIZE = 32 * 32 * 3;

    aug = zeros(size(X));
    for i = 1:size(X, 1)
        img = reshape(X(i, :), 32, 32, 3);
        aug_img = imtranslate(img, [2, 2]);
        aug(i, :) = reshape(aug_img, 1, VEC_SIZE);
    end
end

