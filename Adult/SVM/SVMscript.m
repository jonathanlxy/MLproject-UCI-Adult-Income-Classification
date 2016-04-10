clear
%% Read data
Train = csvread('finalset_cleaned_X.csv', 1, 0);
Test  = csvread('finalset_cleaned_Y.csv', 1, 0);

Xtrain = Train(:, 1:(end-1));
Xtest  = Test (:, 1:(end-1));

Ytrain = Train(:, end);
Ytest  = Test (:, end);

iter = 4000; %[200, 400, 1000, 3000, 3250, 4000, 5000];
Box = 1; %1:10;
Outlier = [0, .001, .005, .01, .05, .1, .5];
Accuracy_Linear = zeros(1, size(iter, 2));
Accuracy_RBF = zeros(1, size(iter, 2));
for v = 1:length(Outlier)
    rng(42)
    SVMModel_Linear = fitcsvm(Xtrain,Ytrain, ...
                       'KernelScale', 'auto', ...
                       'KernelFunction', 'linear', ...
                       'OutlierFraction', Outlier(v), ...
                       'IterationLimit', iter, ...
                       'BoxConstraint', Box);
    prediction_Linear = predict(SVMModel_Linear, Xtest);
    Accuracy_Linear(v) = sum(prediction_Linear == Ytest )/length(Ytest);
    
    rng(42)
    SVMModel_RBF = fitcsvm(Xtrain,Ytrain, ...
                       'KernelScale', 'auto', ...
                       'KernelFunction', 'gaussian', ...
                       'OutlierFraction', Outlier(v), ...
                       'IterationLimit', iter, ...
                       'BoxConstraint', Box);
    prediction_RBF = predict(SVMModel_RBF, Xtest);
    Accuracy_RBF(v) = sum(prediction_RBF == Ytest )/length(Ytest);
end
plot(1:length(Outlier), [Accuracy_Linear; Accuracy_RBF], '-o')
legend('Linear Kernel', 'Non-Linear Kernel', ...
       'Location', 'northoutside')
xlabel('Box Constraint')
ylabel('Accuracy')