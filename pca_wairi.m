function [ Evec, score, EVals, cov_data ] = pca_wairi( data, num )
% PCA function
    % Perform PCA on the given data, and return the principle components, 
    % and, the rotated data, the eigenvalues and the covarience matrix
    
    % center the data
    normed = bsxfun(@minus, data, mean(data, 1));
    % compute the covarience matrix
    cov_data = (normed' * normed) ./ (size(normed, 1) - 1);   
    
    % compute eigenvalues and eigenvectors
    [Evec, EVals]  = eig(cov_data);
    % sort by eigenvalues
    [EVals, idx] = sort(diag(EVals), 'descend');
    Evec = Evec(:, idx);
    % get the principle components
    Evec = Evec(:, 1 : num);

    % just to conform to the convention that the maximal values in each
    % comlumn is positive
    [~,maxind] = max(abs(Evec), [], 1);
    [d1, d2] = size(Evec);
    colsign = sign(Evec(maxind + (0:d1:(d2 - 1) * d1)));
    Evec = bsxfun(@times, Evec, colsign);

    % rotate the data
    score = normed * Evec;
end

