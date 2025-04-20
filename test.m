function saveEvaluationResults(net, augTest, trueLabels, optimizerName, fileName)
    predictedLabels = classify(net, augTest);
    [confMat, ~] = confusionmat(trueLabels, predictedLabels);

    % Hesaplama
    TP = diag(confMat);
    FP = sum(confMat,1)' - TP;
    FN = sum(confMat,2) - TP;

    precision = mean(TP ./ (TP + FP));
    recall = mean(TP ./ (TP + FN));  % Sensitivity
    f1score = 2 * (precision * recall) / (precision + recall);
    accuracy = sum(TP) / sum(confMat(:));

    % Dosyaya Yaz
    fid = fopen(fileName, 'a'); % 'append' modunda aç
    fprintf(fid, '\n=== %s Results ===\n', optimizerName);
    fprintf(fid, 'Accuracy   : %.2f%%\n', accuracy * 100);
    fprintf(fid, 'Precision  : %.2f%%\n', precision * 100);
    fprintf(fid, 'Sensitivity: %.2f%%\n', recall * 100);
    fprintf(fid, 'F1-Score   : %.2f%%\n', f1score * 100);
    fclose(fid);

    % Terminalde de göster
    fprintf('\n=== %s Results ===\n', optimizerName);
    fprintf('Accuracy   : %.2f%%\n', accuracy * 100);
    fprintf('Precision  : %.2f%%\n', precision * 100);
    fprintf('Sensitivity: %.2f%%\n', recall * 100);
    fprintf('F1-Score   : %.2f%%\n', f1score * 100);
end

% Test Etme ve Txt Olarak Kaydetme
trueLabels = imdsTest.Labels;
resultsFile = 'MobileNetv2_High_BrainTumor_Results.txt';

% Dosyayı sıfırla
fid = fopen(resultsFile, 'w');
fprintf(fid, 'Brain Tumor Detection Test Results\n');
fprintf(fid, '=================================\n');
fclose(fid);

% Modeller için hesaplama ve kaydetme
saveEvaluationResults(adamnetwork, augTest, trueLabels, 'Adam', resultsFile);
saveEvaluationResults(rmspropnetwork, augTest, trueLabels, 'RMSProp', resultsFile);
saveEvaluationResults(sgdmnetwork, augTest, trueLabels, 'SGDM', resultsFile);
