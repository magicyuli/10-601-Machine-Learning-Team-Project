function [ a ] = activate( z, type )
    switch type
        case 'SIGMOID'
            a = sigmoid(z);
        case 'ReLU'
            a = max(0, z);
        case 'SOFTMAX'
            a = softmax(z);
        otherwise
            a = z;
    end
end

