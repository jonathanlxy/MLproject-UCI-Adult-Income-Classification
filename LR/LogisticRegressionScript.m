clear
%% Read data
Train = csvread('finalset_cleaned_train.csv', 1, 0);
Test  = csvread('finalset_cleaned_test.csv', 1, 0);

Xtrain = Train(:, 1:(end-1));
Xtest  = Test (:, 1:(end-1));

Ytrain = Train(:, end);
Ytest  = Test (:, end);

%% Set parameters & results containers
% Set iterations
n = 1:30
% Set regularization
lambda = 500

% w container
w_L0 = zeros(length(n), size(Xtrain, 2));
w_L1 = zeros(length(n), size(Xtrain, 2));
w_L2 = zeros(length(n), size(Xtrain, 2));

% Prediction results container
logTestPred_L0  = zeros(size(Test , 1), length(n));
logTestPred_L1  = zeros(size(Test , 1), length(n));
logTestPred_L2  = zeros(size(Test , 1), length(n));

% Accuracy container
TestAccuracy_L0 = zeros(1, length(n));
TestAccuracy_L1 = zeros(1, length(n));
TestAccuracy_L2 = zeros(1, length(n));

% Before iteration one, randomly set w0
rng(42)
% In general, you can generate N random numbers in the interval (a,b) 
% with the formula r = a + (b-a).*rand(N,1)
w0 = -5 + (5 + 5)*rand(1, size(Xtrain, 2));

%% LR training
for i = 1:length(n)
    if i == 1
        newiter = n(i);
        w0_L1 = w0;
        w0_L2 = w0_L1;
    else
        newiter = n(i) - n(i-1);
        % Reuse w from previous iteration
        w0 = w_L0((i-1),:);
        w0_L1 = w_L1((i-1),:);
        w0_L2 = w_L2((i-1),:);
    end
    % Learn w
    w_L0(i,:) = learnLogisticWeights(w0, Xtrain, Ytrain, newiter, 0, lambda);
    w_L1(i,:) = learnLogisticWeights(w0_L1, Xtrain, Ytrain, newiter, 1, lambda);
    w_L2(i,:) = learnLogisticWeights(w0_L2, Xtrain, Ytrain, newiter, 2, lambda);
    % Prediction
    logTestPred_L0 (:, i) = logisticClassify(Xtest , w_L0(i, :));
    logTestPred_L1 (:, i) = logisticClassify(Xtest , w_L1(i, :));
    logTestPred_L2 (:, i) = logisticClassify(Xtest , w_L2(i, :));
    TestAccuracy_L0(i) = sum(logTestPred_L0(:, i) == Ytest )/length(Ytest);
    TestAccuracy_L1(i) = sum(logTestPred_L1(:, i) == Ytest )/length(Ytest);
    TestAccuracy_L2(i) = sum(logTestPred_L2(:, i) == Ytest )/length(Ytest);
end
% TrainAccuracy %View TrainAccuracy
% TestAccuracy %View TestAccuracy

plot(n, [TestAccuracy_L0; TestAccuracy_L1; TestAccuracy_L2])
title({'Figure 3. Change of Prediction Accuracy', ...
       'Step Size = 0.001', ...
        'N = 1:30', ...
       'Lambda = 500'}) %
legend('No Regularization', 'L1 Regularization', 'L2 Regularization', ...
       'Location', 'southeast')
xlabel('Number of learning iterations (n)')
ylabel('Accuracy')