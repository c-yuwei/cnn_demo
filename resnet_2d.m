clc;
clear;

% Load image data, specify the folder path of imd datastare
%data_path = 'D:\melanoma_data\data_store_matlab';
%file_extension = '.jpg';
data_path = 'ich_data';
file_extension = '.dcm';

imds = imageDatastore(data_path, ...
    'IncludeSubfolders',true,'FileExtensions',file_extension,'LabelSource','foldernames');
imds.ReadFcn = @customreader;

% Specify traing and validation sets
numTrainRatio = 0.9;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainRatio,'randomize');

net = resnet50;
%net = mobilenetv2;
%net = inceptionv3;
%net = xception;
%net = vgg16;

lgraph = layerGraph(net);
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:24) = freezeWeights(layers(1:24));
lgraph = createLgraphUsingConnections(layers,connections);

inputSize = layers(1).InputSize;
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
pixelRange = [-3 3];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
%augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
numel(augimdsTrain.Files)
valFrequency = floor(numel(augimdsTrain.Files)/augimdsTrain.MiniBatchSize);


options = trainingOptions('adam', ...
    'InitialLearnRate',0.0001, ...
    'L2Regularization',0.0001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',10, ...
    'MaxEpochs',20, ...
    'MiniBatchSize',32, ...
    'Shuffle','never', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',true, ...
    'Plots','none');

% Training begins...
net = trainNetwork(augimdsTrain,lgraph,options);


% Classify validation set and compute accuracy
YPred = classify(net,augimdsValidation);
YValidation = imdsValidation.Labels;
YPredProb = predict(net,augimdsValidation);

positive_label = 1;
[X,Y,T,AUC] = perfcurve(YValidation,YPredProb(:,2),positive_label);
accuracy = sum(YPred == YValidation)/numel(YValidation)

function data = customreader(filename)
    %data = double(imread(filename));
     datatemp = dicomread(filename);
     data(:,:,1) = datatemp;
     data(:,:,2) = datatemp;
     data(:,:,3) = datatemp;
end
