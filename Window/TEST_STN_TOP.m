clc; clear all;
tic
load('STN_TOP.mat')
Test=imageDatastore('Class','IncludeSubfolders',1,'LabelSource','foldernames');
for j=1:5
Test.Files(end+1:2*(size(Test.Files,1)))=Test.Files;
Test.Labels((size(Test.Labels,1)/2)+1:end)=Test.Labels(1:(size(Test.Labels,1)/2));
end
Test.ReadFcn=@read;
%Проверка
YPred = classify(STN,Test);
%Расчёт точности
accuracy=sum(YPred==Test.Labels)/numel(Test.Labels)
toc