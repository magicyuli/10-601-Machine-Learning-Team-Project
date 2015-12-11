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
batch_sz = 10;

acc_w = zeros(nodes, size(X, 2));
acc_u = zeros(classes, nodes);
acc_b1 = zeros(nodes, 1);
acc_b2 = zeros(classes, 1);

while abs(delta) > 0.0001
    B1=0;
    B2=0;
    iter = iter + 1;
    curRMSE = 0;    
    % Iterate through all examples
    eta = 2 * iter^-0.5;
    % shuffle data
    idx = randperm(n);
    X = X(idx, :);
    Y = Y(idx, :);
    for i = 1 : n
        % Input data from current example set
        I = X(i,:).';
        D = Y(i,:).';

        % Propagate the signals through network
        H = sigmoid(W * I + B1);
        O = sigmoid(U * H + B2);

        % Output layer error
        delta_i = O .* (1 - O) .* (D - O);

        % Calculate error for each node in layer_(n-1)
        delta_j = H .* (1 - H) .* (U.' * delta_i);

        % Adjust weights in matrices sequentially
        acc_u = acc_u + eta .* delta_i * (H.');
        acc_w = acc_w + eta .* delta_j * (I.');
        acc_b1 = acc_b1 + eta .* delta_j;
        acc_b2 = acc_b2 + eta .* delta_i;
        
        curRMSE = curRMSE + sumsqr(D - sigmoid(U * sigmoid(W * I + B1) + B2));        
        
        if mod(n, batch_sz) == 0
            U = U + acc_u + eta * lamda * U / n;
            W = W + acc_w + eta * lamda * W / n;
            B1 = B1 + acc_b1 + eta * lamda * B1 / n;
            B2 = B2 + acc_b2 + eta * lamda * B2 / n;
            acc_w = zeros(nodes, size(X, 2));
            acc_u = zeros(classes, nodes);
            acc_b1 = zeros(nodes, 1);
            acc_b2 = zeros(classes, 1);
        end
        
    end
    disp(curRMSE);
    delta = (curRMSE - prevRMSE) / prevRMSE;
    prevRMSE = curRMSE;
    y = sqrt(curRMSE / n);
    plot(iter, y, '*');
end
end
