% Classify the test set for each model
predictedLabels_adam = classify(adamnetwork, augTest);
predictedLabels_rmsprop = classify(rmspropnetwork, augTest);
predictedLabels_sgdm = classify(sgdmnetwork, augTest);

trueLabels = imdsTest.Labels;

% Compute metrics for each network
computeMetrics(trueLabels, predictedLabels_adam, 'Adam Network');
computeMetrics(trueLabels, predictedLabels_rmsprop, 'RMSProp Network');
computeMetrics(trueLabels, predictedLabels_sgdm, 'SGDM Network');

% Save the trained models
save('Evaluation_Low_trainedCustomCNN_BrainTumor.mat', 'sgdmnetwork','adamnetwork','rmspropnetwork');

% --------------------- Function must be at the end ---------------------
function computeMetrics(trueLabels, predictedLabels, modelName)
    fprintf('\n--- Performance for %s ---\n', modelName);
    
    % Confusion matrix
    confMat = confusionmat(trueLabels, predictedLabels);
    
    % Number of classes
    numClasses = size(confMat, 1);
    
    % Initialize metrics
    precision = zeros(numClasses,1);
    recall = zeros(numClasses,1);
    f1 = zeros(numClasses,1);
    
    for i = 1:numClasses
        TP = confMat(i, i);
        FP = sum(confMat(:, i)) - TP;
        FN = sum(confMat(i, :)) - TP;
        
        precision(i) = TP / (TP + FP + eps);
        recall(i) = TP / (TP + FN + eps);
        f1(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
    end
    
    % Macro-averaged metrics
    avgPrecision = mean(precision);
    avgRecall = mean(recall);
    avgF1 = mean(f1);
    
    % Accuracy
    accuracy = sum(predictedLabels == trueLabels) / numel(trueLabels);
    
    % Display metrics
    fprintf('Accuracy: %.2f%%\n', accuracy * 100);
    fprintf('Precision (macro): %.2f%%\n', avgPrecision * 100);
    fprintf('Sensitivity / Recall (macro): %.2f%%\n', avgRecall * 100);
    fprintf('F1-score (macro): %.2f%%\n', avgF1 * 100);
end
