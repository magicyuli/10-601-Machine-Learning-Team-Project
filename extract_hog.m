function [ hog ] = extract_hog( X, type )
    N = size(X, 1);
    cell_size = 4;
    image_size = 32;
    switch type
        case 'dala'
            data_sz = 36;
        case 'std'
            data_sz = 31;
    end
    hog_size = image_size / cell_size * image_size / cell_size * data_sz;
    hog = zeros(N, hog_size);
    for n = 1 : N
        switch type
            case 'dala'
                hog_tmp = vl_hog(im2single(reshape(X(n, :), 32, 32, 3)),...
                    cell_size, 'variant', 'dalaltriggs', 'BilinearOrientations');
            case 'std'
                hog_tmp = vl_hog(im2single(reshape(X(n, :), 32, 32, 3)),...
                    cell_size);
        end
        hog(n, :) = reshape(hog_tmp, 1, hog_size);
    end
end