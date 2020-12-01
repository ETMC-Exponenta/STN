classdef  TransformImgLayer < nnet.layer.Layer
    
    % —лой выполн€ет аффинное преобразование над входным изображением.
    % ѕараметрами данного сло€ €вл€ютс€:
    %  name-им€ сло€
    %
    % ¬ходами данного сло€ €вл€ютс€ theta-матрица преобразований и img-входной
    % изображение, а выход это преобразованное изображение.
    %
    % “естирование сло€:
    % layer = TransformImgLayer('Transform')
    % checkLayer(layer, {[2 3],[28 28]},'ObservationDimension', 4)
    
    properties (Constant)
        DefaultName = 'TransforImg';
    end
    properties
        DisplayingResults;
    end
    methods
        function layer = TransformImgLayer(name)
            NumInputs=2;
            layer.Name = name;
            layer.NumInputs=NumInputs;
            layer.InputNames{1}='theta';
            layer.InputNames{2}='img';
            layer.OutputNames='OutImg';
        end
        
        function Z = predict(~, theta, img)
            Z=img; out_size=[size(img,1),size(img,2)];
            for i=1:1:size(theta,4)
                Z(:,:,:,i)=transform((theta(:,:,:,i)),(img(:,:,:,i)),out_size);  
            end
            imgvis({theta, img, Z})
        end
    end
end

function  img=transform(theta, im, out_size)
grid(1,:)=reshape(((-1:2/(out_size(1)-1):1)'*ones(1,out_size(2))), 1, []); 
grid(2,:)=reshape((ones(out_size(1),1)*(-1:2/(out_size(2)-1):1)), 1, []); 
grid(3,:)=ones(size(out_size(1)*out_size(2)));
T_g=theta*grid;
x=(T_g(1,:)+1)*(out_size(2))/2;
y=(T_g(2,:)+1)*(out_size(1))/2;
x0=fix(x); x1=x0+1; 
y0=fix(y); y1=y0+1;
x0=C_B_V(x0, out_size(2)); 
x1=C_B_V(x1, out_size(2));
y0=C_B_V(y0, out_size(1)); 
y1=C_B_V(y1, out_size(1));
base=zeros(1,out_size(1)*out_size(2)); 
base_y0=base+y0*out_size(2); 
base_y1=base+y1*out_size(2);
im_flat=(reshape(im, [], 1));
A=(((x1-x).*(y1-y))').*im_flat(base_y0+x0+1);
B=(((x1-x).*(y-y0))').*im_flat(base_y1+x0+1);
C=(((x-x0).*(y1-y))').*im_flat(base_y0+x1+1);
D=(((x-x0).*(y-y0))').*im_flat(base_y1+x1+1);
img=reshape((A+B+C+D),[out_size(1),out_size(2)]);
function x=C_B_V(x,max)
x(x>max-1)=max-1; 
x(x<0)=0; 
end
end

function imgvis(in)
Img=extractdata(in{3}); img=extractdata(in{2}); theta=extractdata(in{1});
subplot(1,2,1);
imshow((img(:,:,:,1)));
title({'theta'});
subplot(1,2,2);
imshow((Img(:,:,:,1)));
title({num2str((theta(:,:,:,1)))});
end