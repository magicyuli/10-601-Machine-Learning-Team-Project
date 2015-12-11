%% Very simple and intuitive neural network implementation
%
%  Carl L?ndahl, 2008
%  email: carl(dot)londahl(at)gmail(dot)com
%  Feel free to redistribute and/or to modify in any way

function [U, W, B1, B2] = NNs_converge(X, Y, nodes, lamda)
% Initialize matrices with random weights 0-1
n = size(X, 1);
classes = size(Y, 2);
W = normrnd(0, 1 / n, nodes, size(X, 2));
U = normrnd(0, 1 / n, classes, nodes);
%bias for input to Hidden

B1 = rand(nodes, 1) - 0.5;
%bias for hidden to output
B2 = rand(classes, 1) - 0.5;

figure; hold on; 
prevRMSE = 0;
delta = 1;
iter = 0;
while abs(delta) > 0.0001
    B1=0;
    B2=0;
    iter = iter + 1;
    curRMSE = 0;    
    % Iterate through all examples
    eta = 2 * iter^-0.5;
    for i = 1 : n
        % Input data from current example set
        I = X(i,:).';
        D = Y(i,:).';

        % Propagate the signals through network
        H = sigmoid(W*I+B1);
        O = sigmoid(U*H+B2);

        % Output layer error
        delta_i = O .* (1 - O) .* (D - O);

        % Calculate error for each node in layer_(n-1)
        delta_j = H .* (1 - H) .* (U.' * delta_i);

        % Adjust weights in matrices sequentially
        U = U + eta .* delta_i * (H.') + eta * lamda * U / n ;
        
        W = W + eta .* delta_j * (I.') + eta * lamda * W / n;
        
        B2 = B2 + eta.*delta_i;
        B1 = B1 + eta.*delta_j;
        curRMSE = curRMSE + sumsqr(D - sigmoid(U * sigmoid(W * I + B1) + B2));        
    end
    disp(curRMSE);
    delta = (curRMSE - prevRMSE) / prevRMSE;
    prevRMSE = curRMSE;
    y = sqrt(curRMSE / n);
    plot(iter, y, '*');
end
end
