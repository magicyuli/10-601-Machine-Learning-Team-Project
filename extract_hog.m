function [ hog ] = extract_hog( X, type )
% Extract HoG features from X based on type
    % There are 2 types of HoGs. One is the standard uncompressed version,
    % another is the richer compressed version (the dalaltriggs variance).
    N = size(X, 1);
    CELL_SIZE = 4;
    IMAGE_SIZE = 32;
    
    switch type
        case 'dala'
            data_sz = 36;
        case 'std'
            data_sz = 31;
    end
    hog_size = IMAGE_SIZE / CELL_SIZE * IMAGE_SIZE / CELL_SIZE * data_sz;
    hog = zeros(N, hog_size);
    for n = 1 : N
        switch type
            case 'dala'
                hog_tmp = vl_hog(im2single(reshape(X(n, :), 32, 32, 3)),...
                    CELL_SIZE, 'variant', 'dalaltriggs', 'BilinearOrientations');
            case 'std'
                hog_tmp = vl_hog(im2single(reshape(X(n, :), 32, 32, 3)),...
                    CELL_SIZE);
        end
        hog(n, :) = reshape(hog_tmp, 1, hog_size);
    end
end