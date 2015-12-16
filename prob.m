function [P] = prob(batch_x, w)
%
%Given some samples, compute the soft max probability
%of they belong to each class
%:param batch_x: some data samples
%:return: a probability matrix
%
        
P = exp(batch_x * w.');
iv_sum = 1 ./ sum(P, 2);
fun = @(A, B) A .* B;
P = bsxfun(fun, P, iv_sum);
end


