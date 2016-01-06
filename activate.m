function [ a ] = activate( z, type )
% Activation function for neural network
    switch type
        case 'SIGMOID'
            a = sigmoid(z);
        case 'ReLU'
            a = max(0, z);
        case 'SOFTMAX'
            a = softmax(z);
        otherwise
            error('Unexpected activation type. Exiting.');
    end
end

