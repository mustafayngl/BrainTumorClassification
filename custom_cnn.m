% Image input size (match dataset preprocessing)
inputSize = [227 227 3];
numClasses = numel(categories(imdsTrain.Labels));

% Augment image data (make sure it's RGB)
augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augVal   = augmentedImageDatastore(inputSize(1:2), imdsVal,   'ColorPreprocessing', 'gray2rgb');
augTest  = augmentedImageDatastore(inputSize(1:2), imdsTest,  'ColorPreprocessing', 'gray2rgb');

% Define a simple custom CNN
layers = [
    imageInputLayer(inputSize, 'Name', 'input')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    globalAveragePooling2dLayer('Name', 'gap')

    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Define training options for each optimizer
options_adam = trainingOptions('adam', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

options_rmsprop = trainingOptions('rmsprop', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

options_sgdm = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the model with each optimizer
adamnetwork = trainNetwork(augTrain, layers, options_adam);
rmspropnetwork = trainNetwork(augTrain, layers, options_rmsprop);
sgdmnetwork = trainNetwork(augTrain, layers, options_sgdm);

% Evaluate model (example: Adam network shown below, repeat for others)
predictedLabels = classify(adamnetwork, augTest);
trueLabels = imdsTest.Labels;
confusionchart(trueLabels, predictedLabels);
accuracy = sum(predictedLabels == trueLabels) / numel(trueLabels);
fprintf('Adam Test Accuracy: %.2f%%\n', accuracy * 100);

predictedLabels_rms = classify(rmspropnetwork, augTest);
accuracy_rms = sum(predictedLabels_rms == trueLabels) / numel(trueLabels);
fprintf('RMSprop Accuracy: %.2f%%\n', accuracy_rms * 100);

predictedLabels_sgdm = classify(sgdmnetwork, augTest);
accuracy_sgdm = sum(predictedLabels_sgdm == trueLabels) / numel(trueLabels);
fprintf('SGDM Accuracy: %.2f%%\n', accuracy_sgdm * 100);


% Save all networks
save('Low_trainedCustomCNN_BrainTumor.mat', 'adamnetwork', 'rmspropnetwork', 'sgdmnetwork');
