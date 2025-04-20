% Get file paths in valData
valFiles = valData.Files;

% Get all labels from the original image datastore
allLabels = imds.Labels;
allFiles = imds.Files;

% Find indices of valData files in the original imds
[~, idx] = ismember(valFiles, allFiles);

% Get corresponding true labels
trueLabels = allLabels(idx);

% Loop through each trained network and display its confusion matrix
for i = 1:6
    % Get the trained network
    net = eval(sprintf('trainedNetwork_%d', i));

    % Predict labels using the network
    predictedLabels = classify(net, valData);

    % Display the confusion matrix
    figure;
    confusionchart(trueLabels, predictedLabels);
    title(sprintf('Confusion Matrix for trainedNetwork_%d', i));
end
