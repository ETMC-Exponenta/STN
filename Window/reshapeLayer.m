classdef reshapeLayer < nnet.layer.Layer
    
    % Слой выполняет функцию reshape-данная функция переформирует массив,
    % входные параметры, для данного слоя описаны ниже:
    %   name-имя слоя, по умолчанию 'reshape'
    %
    % Тест слоя
    %   layer = reshapeLayer("reshape4",[1,1,80])
    %   checkLayer(layer, [2,2,20,128],'ObservationDimension', 4)
    
    properties (Constant)
        DefaultName = 'reshape'
    end
    properties
        SizeOut=[];
    end
    
    methods
        function layer = reshapeLayer(name, SizeOut)%Объявление
            layer.Name = name;
            layer.SizeOut=SizeOut;
        end
        
        function Z = predict(layer, varargin)% Во время обучения
            S=layer.SizeOut;
            for i=1:1:size(varargin{1},4)
                Z(1:S(1),1:S(2),1:S(3),i)=reshape(varargin{1}(:,:,:,i),S(1),S(2),S(3));
            end
        end
    end
end