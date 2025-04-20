
% Classify the test set
predictedLabels = classify(sgdmnetwork, augTest);
trueLabels = imdsTest.Labels;

% Plot confusion matrix and calculate accuracy
confusionchart(trueLabels, predictedLabels);
accuracy = sum(predictedLabels == trueLabels) / numel(trueLabels);
fprintf('Test Accuracy2f%%\n', accuracy * 100);