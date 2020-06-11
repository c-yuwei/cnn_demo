% Load image data, specify the folder path of imd datastare
clc; clear;
data_path = 'ich_data';
file_extension = '.dcm';
%data_path = 'melanoma_data';
%file_extension = '.jpg';
imds = imageDatastore(data_path, ...
    'IncludeSubfolders',true,'FileExtensions',file_extension,'LabelSource','foldernames');
imds.ReadFcn = @customreader;

% Specify traing and validation sets
numTrainRatio = 0.9;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainRatio,'randomize');

% Define neural network compositions
layers = [
    %imageInputLayer([450 600 3],'Normalization','zscore') 
    imageInputLayer([512 512 1],'Normalization','zscore') 
    
    convolution2dLayer(3,8,'Padding','same') 
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
   
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same') 
    batchNormalizationLayer
    reluLayer
      
    fullyConnectedLayer(128)
    dropoutLayer(0.5)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

% Specify traing options
miniBatchSize = 64;
valFrequency = floor(numel(imdsTrain.Files)/miniBatchSize);
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'L2Regularization',0.0002, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',20, ...
    'MaxEpochs',40, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','once', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',true, ...
    'ExecutionEnvironment','auto', ...
    'Plots','none');

% Training begins...
net = trainNetwork(imdsTrain,layers,options);


% Classify validation set and compute accuracy
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
YPredProb = predict(net,imdsValidation);

accuracy = sum(YPred == YValidation)/numel(YValidation)

positive_label = 1;
[X,Y,T,AUC] = perfcurve(YValidation,YPredProb(:,2),positive_label);

function data = customreader(filename)
    %data = imread(filename);
    data = dicomread(filename);
end
