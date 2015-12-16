function [ Model ] = NNs_converge(X, Y, config, val_data, val_label, Evec)
% --------------------- START CONFIGURATION ------------------------ %
    ACTIVATION_TYPE = 'ReLU';
%     ACTIVATION_TYPE = 'SIGMOID';
    
%     OUTPUT_ACTIVATION_TYPE = 'SIGMOID';
    OUTPUT_ACTIVATION_TYPE = 'SOFTMAX';
    
%     ERROR_TYPE = 'SQUARE';
%     ERROR_TYPE = 'CROSS_ENTROPY';
    ERROR_TYPE = 'NEGATIVE_LOG_LIKELIHOOD';
    
    BATCH_SIZE = 2;
% --------------------- END CONFIGURATION ------------------------ %

% --------------------- START INITILIZATION ------------------------ %
    % Initialize matrices with random weights 0-1
    N = size(X, 1);
    W = cell(1, config.n_layers - 1);
    B = cell(1, config.n_layers - 1);
    W_V = cell(1, config.n_layers - 1);
    B_V = cell(1, config.n_layers - 1);
    Z = cell(1, config.n_layers - 1);
    A = cell(1, config.n_layers - 1);
    DELTA = cell(1, config.n_layers - 1);
    for i = 1:config.n_layers - 1
        W{i} = randn(config.n_nodes(i + 1), config.n_nodes(i)) * sqrt(2 ./ config.n_nodes(i));
        B{i} = ones(config.n_nodes(i + 1), 1) * 0.01;
        W_V{i} = zeros(config.n_nodes(i + 1), config.n_nodes(i));
        B_V{i} = zeros(config.n_nodes(i + 1), 1);
        Z{i} = zeros(config.n_nodes(i + 1), BATCH_SIZE);
        A{i} = zeros(config.n_nodes(i + 1), BATCH_SIZE);
        DELTA{i} = zeros(config.n_nodes(i + 1), BATCH_SIZE);
    end

    figure; hold on;
    
    prev_err = 0;
    delta_err = 1;
    iter = 0;
    epoch = 0;
    mu = 0.9;
    batch_num = N / BATCH_SIZE;
    best_accu = 0;
% --------------------- END INITILIZATION ------------------------ %

    while ~(delta_err < 0 && delta_err > -1e-5)
%     while epoch < 14
        epoch = epoch + 1;
        err = 0;
        % shuffle data
        idx = randperm(N);
        X = X(idx, :);
        Y = Y(idx, :);
        for i = 1 : batch_num
            % upate the learning rate
            iter = iter + 1;
            switch ERROR_TYPE
                case 'CROSS_ENTROPY'
                    eta = 0.5 * iter^-0.05;
                case 'NEGATIVE_LOG_LIKELIHOOD'
                    eta = 0.001;
                case 'SQUARE'
                    eta = 2 * iter^-0.05;
            end
                        
            % get a mini-batch
            head = (i - 1) * BATCH_SIZE + 1;
            tail = i * BATCH_SIZE;
            a0 = X(head:tail,:).';
            y = Y(head:tail,:).';

            % ---------------Forward--------------- %
            act = a0;
            for l = 1:config.n_layers - 2
                Z{l} = bsxfun(@plus, W{l} * act, B{l});
                A{l} = activate(Z{l}, ACTIVATION_TYPE);
                act = A{l};
            end
            Z{config.n_layers - 1} = bsxfun(@plus, W{config.n_layers - 1} * act, B{config.n_layers - 1});
            A{config.n_layers - 1} = activate(Z{config.n_layers - 1}, OUTPUT_ACTIVATION_TYPE);
            out = A{config.n_layers - 1};

            % ---------------Backward--------------- %
            % Compute delta's
            switch OUTPUT_ACTIVATION_TYPE
                case 'SIGMOID'
                    DELTA{config.n_layers - 1} = out .* (1 - out) .* (out - y);
                case 'SOFTMAX'
                    DELTA{config.n_layers - 1} = out - y;
            end
            
            switch ACTIVATION_TYPE
                case 'SIGMOID'
                    for j = config.n_layers - 2:-1:1
                        DELTA{j} = A{j} .* (1 - A{j}) .* (W{j + 1}.' * DELTA{j + 1});
                    end
                case 'ReLU'
                    for j = config.n_layers - 2:-1:1
                        DELTA{j} = double(Z{j} > 0) .* (W{j + 1}.' * DELTA{j + 1});
                    end
            end
            
            % ---------------Update Weights--------------- %
            % Adjust weights in matrices sequentially
            for j = config.n_layers - 1:-1:2
                W_V{j} = mu * W_V{j} - eta .* (DELTA{j} * A{j - 1}' + config.lambda * W{j});
                W{j} = W{j} + W_V{j};
                B_V{j} = mu * B_V{j} - eta .* (sum(DELTA{j}, 2) + config.lambda * B{j});
                B{j} = B{j} + B_V{j};
            end
            W_V{1} = mu * W_V{1} - eta .* (DELTA{1} * a0' + config.lambda * W{1});
            W{1} = W{1} + W_V{1};
            B_V{1} = mu * B_V{1} - eta .* (sum(DELTA{1}, 2) + config.lambda * B{1});
            B{1} = B{1} + B_V{1};

            % ---------------Compute Error--------------- %
            act = a0;
            for l = 1:config.n_layers - 2
                Z{l} = bsxfun(@plus, W{l} * act, B{l});
                A{l} = activate(Z{l}, ACTIVATION_TYPE);
                act = A{l};
            end
            Z{config.n_layers - 1} = bsxfun(@plus, W{config.n_layers - 1} * act, B{config.n_layers - 1});
            A{config.n_layers - 1} = activate(Z{config.n_layers - 1}, OUTPUT_ACTIVATION_TYPE);
            out = A{config.n_layers - 1};
            
            switch ERROR_TYPE
                case 'CROSS_ENTROPY'
                    % smoother: 1e-8
                    err = err - sum(sum(y .* log(max(1e-8, out)) + (1 - y) .* log(max(1 - out, 1e-8))));
                case 'NEGATIVE_LOG_LIKELIHOOD'
                    err = err - sum(sum(y .* log(max(1e-8, out))));
                case 'SQUARE'
                    err = err + 0.5 * sumsqr(y - out);
            end
        end
        err = err / N;
        if prev_err > 0
            delta_err = (err - prev_err) / prev_err;
        end
        prev_err = err;
        fprintf('epoch: %d,  error: %f,  delta: %f,  eta: %f\n', epoch, err, delta_err, eta);
        M = struct('n_layers', config.n_layers - 1, 'W', W, 'B', B,...
            'act_t', ACTIVATION_TYPE, 'out_act_t', OUTPUT_ACTIVATION_TYPE);
        Model = struct('M', M, 'Evec', Evec);
        accu = sum(predict_NN(Model, val_data) == val_label)/size(val_data, 1);
        fprintf('accu: %f\n', accu);
        plot(epoch, accu, '*');
        if accu > 0.6 && accu >= best_accu
            best_accu = accu;
            save('Model4.mat', 'Model');
        end
    end
    Model = struct('n_layers', config.n_layers - 1, 'W', W, 'B', B,...
        'act_t', ACTIVATION_TYPE, 'out_act_t', OUTPUT_ACTIVATION_TYPE);
end
