function [ model ] = train( X, Y )
    %Output the model given the data input
    %   Train a model using the data based on the classifier seleted

    %%%%% start training %%%%%
    type = 'SVM';
    %type = 'NN';
    %type = 'LR';

    Y = double(Y);
    switch type
        case 'SVM'
            model = train_svm(X, Y);
        case 'NN'

        case 'LR'

        otherwise
            error('Unexpected classifier type. Exiting.');
    end

end

