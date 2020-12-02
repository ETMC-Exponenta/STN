classdef reshapeLayer < nnet.layer.Layer
    
    % Слой выполняет функцию reshape-данная функция переформирует массив,
    % входные параметры, для данного слоя описаны ниже:
    %   name-имя слоя, по умолчанию 'reshape'
    %
    % Тест слоя
    %   layer = reshapeLayer("reshape",[1,1,80])
    %   checkLayer(layer, [2,2,20,128],'ObservationDimension', 4)
    
    properties (Constant)
        DefaultName = 'reshape'
    end
    properties
        SizeOut=[];
    end
    
    methods
        function layer = reshapeLayer(name, SizeOut)
            layer.Name = name;
            layer.SizeOut=SizeOut;
        end
        
        function Z = predict(layer, varargin)
            Z=reshape(varargin{1},layer.SizeOut(1),layer.SizeOut(2),layer.SizeOut(3),[]);
        end
    end
end