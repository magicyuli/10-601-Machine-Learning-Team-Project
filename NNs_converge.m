%% Very simple and intuitive neural network implementation
%
%  Carl L?ndahl, 2008
%  email: carl(dot)londahl(at)gmail(dot)com
%  Feel free to redistribute and/or to modify in any way

function [U, W] = NNs_converge(X, Y, nodes, eta)
% Initialize matrices with random weights 0-1
n = size(X, 1);
W = normrnd(0, 1 / n, nodes, size(X, 2));
U = normrnd(0, 1 / n, size(Y, 2), nodes);

figure; hold on; 
prevRMSE = 0;
delta = 1;
iter = 0;
while abs(delta) > 0.00001
    iter = iter + 1;
    curRMSE = 0;    
    % Iterate through all examples
    for i = 1 : n
        % Input data from current example set
        I = X(i,:).';
        D = Y(i,:).';

        % Propagate the signals through network
        H = f(W*I);
        O = f(U*H);

        % Output layer error
        delta_i = O .* (1 - O) .* (D - O);

        % Calculate error for each node in layer_(n-1)
        delta_j = H .* (1 - H) .* (U.' * delta_i);

        % Adjust weights in matrices sequentially
        U = U + eta .* delta_i * (H.');
        W = W + eta .* delta_j * (I.');
        
        curRMSE = curRMSE + norm(D - f(U * f(W * I)), 2);        
    end
    disp(curRMSE);
    delta = (curRMSE - prevRMSE) / prevRMSE;
    prevRMSE = curRMSE;
    y = curRMSE / n;
    plot(iter, y, '*');
end
end


function x = f(x)
x = 1./(1+exp(-x));
end