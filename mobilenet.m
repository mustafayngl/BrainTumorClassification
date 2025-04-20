clc;

% === Dataset Path ve Image Boyutu ===
datasetPath = 'C:\Users\mmy\Downloads\Brain Tumor DL\brain_tumor_dataset';
imageSize = [224 224 3]; % MobileNetV2 için 224x224 RGB

% === ImageDatastore ve Bölme ===
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

[imdsTrain, imdsVal, imdsTest] = splitEachLabel(imds, 0.7, 0.15, 0.15, 'randomized');

% === Augmented Image Datastore ===
augTrain = augmentedImageDatastore(imageSize, imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augVal   = augmentedImageDatastore(imageSize, imdsVal,   'ColorPreprocessing', 'gray2rgb');
augTest  = augmentedImageDatastore(imageSize, imdsTest,  'ColorPreprocessing', 'gray2rgb');

disp("Detected Classes:");
disp(categories(imds.Labels));

% === MobileNetV2 Ağını Yükle ===
net = mobilenetv2;
lgraph = layerGraph(net);

numClasses = numel(categories(imdsTrain.Labels));

% === Son Katmanları Kaldır ===
layersToRemove = {'Logits','Logits_softmax','ClassificationLayer_Logits'};
lgraph = removeLayers(lgraph, layersToRemove);

% === Yeni Katmanları Ekle ===
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'new_softmax')
    classificationLayer('Name', 'new_output')
];

% === Yeni Katmanları Bağla ===
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'global_average_pooling2d_1', 'new_fc');

% === Ağ Yapısını Görüntüle ===
analyzeNetwork(lgraph);

% === Eğitim Ayarları ===
options_adam = trainingOptions('adam', ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

options_rmsprop = trainingOptions('rmsprop', ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

options_sgdm = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% === Ağları Eğit ===
adamnetwork = trainNetwork(augTrain, lgraph, options_adam);
rmspropnetwork = trainNetwork(augTrain, lgraph, options_rmsprop);
sgdmnetwork = trainNetwork(augTrain, lgraph, options_sgdm);

% === Test ve Performans Hesaplama Fonksiyonu ===
function evaluateNetwork(net, augTest, trueLabels, optimizerName)
    predictedLabels = classify(net, augTest);

    % Confusion Matrix
    figure;
    cm = confusionchart(trueLabels, predictedLabels);
    cm.Title = sprintf('%s Confusion Matrix', optimizerName);

    % Numeric Confusion Matrix
    [confMat, order] = confusionmat(trueLabels, predictedLabels);

    % Metrics Calculation
    TP = diag(confMat);
    FP = sum(confMat,1)' - TP;
    FN = sum(confMat,2) - TP;
    TN = sum(confMat(:)) - TP - FP - FN;

    precision = mean(TP ./ (TP + FP));
    recall = mean(TP ./ (TP + FN));  % Sensitivity
    f1score = 2 * (precision * recall) / (precision + recall);
    accuracy = sum(TP) / sum(confMat(:));

    % Results
    fprintf('\n=== %s Results ===\n', optimizerName);
    fprintf('Accuracy   : %.2f%%\n', accuracy * 100);
    fprintf('Precision  : %.2f%%\n', precision * 100);
    fprintf('Sensitivity: %.2f%%\n', recall * 100);
    fprintf('F1-Score   : %.2f%%\n', f1score * 100);
end

% === Test Set Evaluation ===
trueLabels = imdsTest.Labels;

evaluateNetwork(adamnetwork, augTest, trueLabels, 'Adam');
evaluateNetwork(rmspropnetwork, augTest, trueLabels, 'RMSProp');
evaluateNetwork(sgdmnetwork, augTest, trueLabels, 'SGDM');

% === Modelleri Kaydet ===
save('Trained_High_MobileNetV2_BrainTumor.mat', 'adamnetwork', 'rmspropnetwork', 'sgdmnetwork');
