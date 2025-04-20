imds = imageDatastore('C:\Users\mmy\Downloads\Brain Tumor DL\brain_tumor_dataset', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

[imdsTrain, imdsRest] = splitEachLabel(imds, 0.7, 'randomized');
[imdsVal, imdsTest] = splitEachLabel(imdsRest, 0.5, 'randomized');
inputSize = [227 227];
augTrain = augmentedImageDatastore([227 227], imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augVal   = augmentedImageDatastore([227 227], imdsVal,   'ColorPreprocessing', 'gray2rgb');
augTest  = augmentedImageDatastore([227 227], imdsTest,  'ColorPreprocessing', 'gray2rgb');
