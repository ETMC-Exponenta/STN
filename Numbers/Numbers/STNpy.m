clc; clear all;
%% Загрузите данные как ImageDatastore объект.
imdsTrain = processMNISTimages('train-images.idx3-ubyte');
LabelsTrain = processMNISTlabels('train-labels.idx1-ubyte');
imdsTest = processMNISTimages('t10k-images.idx3-ubyte');
LabelsTest = processMNISTlabels( 't10k-labels.idx1-ubyte');

%% Объявление сети STN
STN = layerGraph();
stn = imageInputLayer([28 28 1],"Name","input");
STN = addLayers(STN,stn);

stn = [...
    %localization-network
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
    reshapeLayer("reshape2",[2,3,1])]; % ME
STN = addLayers(STN,stn);

stn = [...
    TransformImgLayer('Transform') 
    %CNN
    convolution2dLayer([5 5],10,"Name","conv3")
    maxPooling2dLayer(2,"Name","max3")
    reluLayer("Name","ReLu4")
    convolution2dLayer([5 5],20,"Name","conv4")
    dropoutLayer("Name","dropout1")
    maxPooling2dLayer(2,"Name","max4")
    reluLayer("Name","ReLu5")
    reshapeLayer("reshape4",[1,18*18*20,1]) 
    fullyConnectedLayer(50,"Name","fc3")
    reluLayer("Name","ReLu6")
    dropoutLayer("Name","dropout2")
    fullyConnectedLayer(10,"Name","fc4")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
STN = addLayers(STN,stn);

STN = connectLayers(STN,"input","conv1");
STN = connectLayers(STN,"input","Transform/img");
STN = connectLayers(STN,"reshape2","Transform/theta");
clear stn; 
%% Обучение и тестирование сети STN
%Анализ сети
%analyzeNetwork(STN);

% Опции для обучения 
options = trainingOptions('sgdm', ... 
    'MaxEpochs',20,...
    'MiniBatchSize',64, ...
    'InitialLearnRate',0.01, ...
    'Plots','training-progress');
%Обучение
STN = trainNetwork(imdsTrain,LabelsTrain,STN,options);
%Проверка
YPred = classify(STN,imdsTest);
%Расчёт точности
accuracy = sum(YPred==LabelsTest)/numel(LabelsTest);
%Визуализация случайного изображения из тестовой выборки
% i=randi(10000);
% YPred = classify(STN,imdsTest(:,:,:,i));
% disp(YPred);

function X = processMNISTimages(filename)
[fileID,errmsg] = fopen(filename,'r','b');
if fileID < 0
    error(errmsg);
end
magicNum = fread(fileID,1,'int32',0,'b');
numImages = fread(fileID,1,'int32',0,'b');
numRows = fread(fileID,1,'int32',0,'b');
numCols = fread(fileID,1,'int32',0,'b');
X = fread(fileID,inf,'unsigned char');
X = reshape(X,numCols,numRows,numImages);
X = permute(X,[2 1 3]); X = X./255;
X = reshape(X, [28,28,1,size(X,3)]);
X = dlarray(X, 'SSCB'); fclose(fileID);
end

function Y = processMNISTlabels(filename)
[fileID,errmsg] = fopen(filename,'r','b');
if fileID < 0
    error(errmsg);
end
magicNum = fread(fileID,1,'int32',0,'b');
numItems = fread(fileID,1,'int32',0,'b');
Y = fread(fileID,inf,'unsigned char');
Y = categorical(Y); fclose(fileID);
end