function [ Evec, score, EVals, cov_data ] = pca_wairi( data, num )
%PCA Summary of this function goes here
%   Detailed explanation goes here
%[coeff,score,latent,tsquared,explained,mu] = pca(hog, 'Algorithm', 'eig', 'NumComponents', 500); 
% Xi - mean(X(i,;))
normed = bsxfun(@minus, data, mean(data, 1));


cov_data = (normed'*normed)./ (size(normed,1) - 1);   

 
[Evec, EVals]  = eig(cov_data);    
[EVals, idx] = sort(diag(EVals), 'descend');
Evec = Evec(:, idx);
Evec = Evec(:, 1 : num);

[~,maxind] = max(abs(Evec), [], 1);
[d1, d2] = size(Evec);
colsign = sign(Evec(maxind + (0:d1:(d2-1)*d1)));
Evec = bsxfun(@times, Evec, colsign);

score = normed * Evec;
end

