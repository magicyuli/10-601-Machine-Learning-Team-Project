function [ X, Y ] = augment( X, Y )
% Return the augmentation set for X
    % The augmentation is done by flipping the images
    
    VEC_SIZE = 32 * 32 * 3;

    aug = zeros(size(X));
    for i = 1:size(X, 1)
        img = reshape(X(i, :), 32, 32, 3);
        aug_img = flip(img, 1);
        aug(i, :) = reshape(aug_img, 1, VEC_SIZE);
    end
    X = [X; aug];
    Y = [Y; Y];
end

