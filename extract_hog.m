function [ hog ] = extract_hog( X )
    N = size(X, 1);
    cell_size = 4;
    image_size = 32;
    hog_size = image_size / cell_size * image_size / cell_size * 36;
    hog = zeros(N, hog_size);
    for n = 1 : N
        hog_tmp = vl_hog(im2single(reshape(X(n, :), 32, 32, 3)), cell_size, 'variant', 'dalaltriggs', 'BilinearOrientations');
        hog(n, :) = reshape(hog_tmp, 1, hog_size);
    end
end