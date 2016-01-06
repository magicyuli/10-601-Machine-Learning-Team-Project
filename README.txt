10-601B CIFAR-10 Team Project README
Team: WAIRI
Members: yilongc, liruoyay

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
HOW TO TRAIN:

%% Train SVM
Model = train(X,Y);

%% Train NN
Model1 = train1(X,Y);

%% Train k-NN
Model2 = train2(X,Y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
HOW TO PREDICT:

%% Predict SVM
y_test = classify(Model, X_test);

%% Predict NN
y_test1 = classify1(Model1, X_test);

%% Predict k-NN
y_test2 = classify2(Model2, X_test);
