clc; clear all;
%% Загрузите данные как ImageDatastore объект.
Train=imageDatastore('Class','IncludeSubfolders',1,'LabelSource','foldernames');
Test=imageDatastore('Class','IncludeSubfolders',1,'LabelSource','foldernames');
for j=1:7
Train.Files(end+1:2*(size(Train.Files,1)))=Train.Files;
Train.Labels((size(Train.Labels,1)/2)+1:end)=Train.Labels(1:(size(Train.Labels,1)/2));
if j<3
Test.Files(end+1:2*(size(Test.Files,1)))=Test.Files;
Test.Labels((size(Test.Labels,1)/2)+1:end)=Test.Labels(1:(size(Test.Labels,1)/2));
end
end
Train.ReadFcn=@read; Test.ReadFcn=@read;
%% Объявление сети STN
lgraph = layerGraph();
tempLayers = imageInputLayer([50 50 3],"Name","input","Normalization","none");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [...
    dropoutLayer(0.2,"Name","dropout")
    %localization-network
    NormLayer("Norm")
    convolution2dLayer([7,7],8,"Name","conv1")
    maxPooling2dLayer(2,"Name","max1","Stride",2)
    reluLayer("Name","ReLu1")
    convolution2dLayer([5,5],10,"Name","conv2")
    maxPooling2dLayer(2,"Name","max2","Stride",2)
    reluLayer("Name","ReLu2")
    %Regressor
    fullyConnectedLayer(32,"Name","fc1")
    reluLayer("Name","ReLu3")
    fullyConnectedLayer(6,"Name","fc2")
    reshapeLayer("reshape",[2,3,1])]; 
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    TransformImgLayer("Transform")
    %CNN
    convolution2dLayer([3 3],8,"Name","conv3")
    batchNormalizationLayer("Name","Batch1")
    reluLayer("Name","ReLu4")
    maxPooling2dLayer([2 2],"Name","max3","Stride",[2 2])
    convolution2dLayer([3 3],16,"Name","conv4")
    batchNormalizationLayer("Name","Batch2")
    reluLayer("Name","ReLu5")
    maxPooling2dLayer([2 2],"Name","max4","Stride",[2 2])
    convolution2dLayer([3 3],32,"Name","conv5")
    batchNormalizationLayer("Name","Batch3")
    reluLayer("Name","ReLu6")
    fullyConnectedLayer(99,"Name","fc3")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","Class")];
lgraph = addLayers(lgraph,tempLayers);

clear tempLayers;

lgraph = connectLayers(lgraph,"input","dropout");
lgraph = connectLayers(lgraph,"input","Transform/img");
lgraph = connectLayers(lgraph,"reshape","Transform/theta");
%% Обучение и тестирование сети STN
%Анализ сети
% analyzeNetwork(lgraph);
% Опции для обучения 
options = trainingOptions('sgdm', ... 
    'MaxEpochs',20,...
    'MiniBatchSize',64, ...
    'InitialLearnRate',0.01, ...
    'Shuffle', 'never', ...
    'Plots','training-progress');
%Обучение
STN = trainNetwork(Train,lgraph,options);
%Проверка
YPred = classify(STN,Test);
%Расчёт точности
accuracy = sum(YPred==Test.Labels)/numel(Test.Labels)