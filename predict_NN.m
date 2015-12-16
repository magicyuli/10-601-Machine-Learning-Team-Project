function [indices] = predict_NN(Model, X)
    test_hog = extract_hog(X, 'dala');
    test_hog = bsxfun(@minus, test_hog, mean(test_hog, 1)) * Model.Evec;
    test_hog = test_hog';
    
    ACTIVATION_TYPE = Model.M(1).act_t;
    OUTPUT_ACTIVATION_TYPE = Model.M(1).out_act_t;
    n_layers = Model.M(1).n_layers;

    act = test_hog;
    for l = 1:n_layers - 1
        z = bsxfun(@plus, Model.M(l).W * act, Model.M(l).B);
        act = activate(z, ACTIVATION_TYPE);
    end
    z = bsxfun(@plus, Model.M(n_layers).W * act, Model.M(n_layers).B);
    out = activate(z, OUTPUT_ACTIVATION_TYPE);

    [~, indices] = max(out);
    indices = (indices - 1)';

end
