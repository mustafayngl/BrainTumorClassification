clc;

% Dataset path
datasetPath = 'C:\Users\mmy\Downloads\Brain Tumor DL\brain_tumor_dataset';

% MobileNetV2 image input size
imageSize = [224 224 3];

% Image datastore oluştur
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Veri setini train / validation / test olarak ayır
[imdsTrain, imdsVal, imdsTest] = splitEachLabel(imds, 0.7, 0.15, 0.15, 'randomized');

% Augmented datastore — grayscale bile olsa MobileNet RGB ister!
augTrain = augmentedImageDatastore(imageSize, imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augVal   = augmentedImageDatastore(imageSize, imdsVal,   'ColorPreprocessing', 'gray2rgb');
augTest  = augmentedImageDatastore(imageSize, imdsTest,  'ColorPreprocessing', 'gray2rgb');

% Sınıf isimlerini göster
disp("Detected Classes:");
disp(categories(imds.Labels));
