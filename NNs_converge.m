function [w2, w1, b1, b2] = NNs_converge(X, Y, nodes, lamda)
%     ACTIVATION_TYPE = 'ReLU';
    ACTIVATION_TYPE = 'SIGMOID';
    
    OUTPUT_ACTIVATION_TYPE = 'SIGMOID';
%     OUTPUT_ACTIVATION_TYPE = 'SOFTMAX';
    
    ERROR_TYPE = 'SQUARE';
%     ERROR_TYPE = 'CROSS_ENTROPY';
%     ERROR_TYPE = 'NEGATIVE_LOG_LIKELIHOOD';
    
    BATCH_SIZE = 10;

    % Initialize matrices with random weights 0-1
    n = size(X, 1);
    classes = size(Y, 2);
    
    w1 = normrnd(0, 1 / n, nodes, size(X, 2));
    w2 = normrnd(0, 1 / n, classes, nodes);
    b1 = normrnd(0, 1 / n, nodes, 1);
    b2 = normrnd(0, 1 / n, classes, 1);

    figure; hold on; 
    prev_err = 0;
    delta = 1;
    iter = 0;
    epoch = 0;
    batch_num = n / BATCH_SIZE;

    while abs(delta) > 1e-4
        epoch = epoch + 1;
        err = 0;
        % shuffle data
        idx = randperm(n);
        X = X(idx, :);
        Y = Y(idx, :);
        for i = 1 : batch_num
            % upate learning rate
            iter = iter + 1;
            eta = 2 * iter^-0.5;
            
            % get a mini-batch
            head = (i - 1) * BATCH_SIZE + 1;
            tail = i * BATCH_SIZE;
            a1 = X(head:tail,:).';
            y = Y(head:tail,:).';

            % ---------------Forward--------------- %
            z2 = bsxfun(@plus, w1 * a1, b1);
            a2 = activate(z2, ACTIVATION_TYPE);
            z3 = bsxfun(@plus, w2 * a2, b2);
            a3 = activate(z3, OUTPUT_ACTIVATION_TYPE);

            % ---------------Backward--------------- %
            % Compute delta's
            switch OUTPUT_ACTIVATION_TYPE
                case 'SIGMOID'
                    delta_3 = a3 .* (1 - a3) .* (a3 - y);
                case 'SOFTMAX'
                    delta_3 = a3 - y;
            end
            
            switch ACTIVATION_TYPE
                case 'SIGMOID'
                    delta_2 = a2 .* (1 - a2) .* (w2.' * delta_3);
                case 'ReLU'
                    delta_2 = double(z2 > 0) .* (w2.' * delta_3);
            end
            
            % ---------------Update Weights--------------- %
            % Adjust weights in matrices sequentially
            w2 = w2 - eta .* (delta_3 * (a2.') + lamda * w2 / n);
            w1 = w1 - eta .* (delta_2 * (a1.') + lamda * w1 / n);
            b2 = b2 - eta .* (sum(delta_3, 2) + lamda * b2 / n);
            b1 = b1 - eta .* (sum(delta_2, 2) + lamda * b1 / n);

            % ---------------Compute Error--------------- %
            switch ERROR_TYPE
                case 'CROSS_ENTROPY'
                    err = err - sum(sum(y .* log(a3) + (1 - y) .* log(1 - a3)));
                case 'NEGATIVE_LOG_LIKELIHOOD'
                    err = err - sum(sum(y .* log(a3)));
                case 'SQUARE'
                    err = err + 0.5 * sumsqr(y - a3);
            end
        end
        err = err / n;
        fprintf('epoch: %d, error: %f\n', epoch, err);
        if prev_err > 0
            delta = (err - prev_err) / prev_err;
        end
        prev_err = err;
        plot(epoch, err, '*');
    end
end

function [ a ] = activate( z, type )
    switch type
        case 'SIGMOID'
            a = sigmoid(z);
        case 'ReLU'
            a = max(0, z);
        otherwise
            a = z;
    end
end
