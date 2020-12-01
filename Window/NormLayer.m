classdef  NormLayer < nnet.layer.Layer
    
    %Слой выполняет нормализацию входных данных для нейронной сети STN
    % входные параметры, для данного слоя описаны ниже:
    % name-имя слоя, по умолчанию 'NormLayer'
    %
    % Тестирование слоя:
    % layer = NormLayer('NORM')
    % checkLayer(layer, {[28 28]},'ObservationDimension', 4)
    
    properties (Constant)
        DefaultName = 'NormLayer';
    end
    methods
        function layer = NormLayer(name)
            NumInputs=1;
            layer.Name = name;
            layer.NumInputs=NumInputs;
        end
        function Z = predict(~, img)
            for i=1:1:size(img,4)
                im = C_B_V(fix(img(:,:,1,i)))./255;
                im(1:end,1)=0; im(1:end,end)=0; 
                im(1,1:end)=0; im(end,1:end)=0; 
                im(im<0.3)=0; im(im>0.75)=0;
                Z(:,:,:,i)=im;
            end
        end
    end
end
function x=C_B_V(x)
x(x>255)=255; 
x(x<0)=0; 
end