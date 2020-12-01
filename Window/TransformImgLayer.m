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
    properties (Learnable)
        Weights
    end
    methods
        
        function layer = TransformImgLayer(name)
            layer.Weights = [2, 2, 0; 2, 2, 0];
            NumInputs=2;
            layer.Name = name;
            layer.NumInputs=NumInputs;
            layer.InputNames{1}='theta';
            layer.InputNames{2}='img';
            layer.OutputNames='OutImg';
        end
        
        function Z = predict(layer, theta, img)
            Z=img; s1=dlarray(size(img,1));
            theta=theta.*layer.Weights;
            out_size=[s1,s1];
            for i=1:dlarray(size(theta,4))
                for j=1:dlarray(size(img,3))
                    Z(:,:,j,i)=transform((theta(:,:,:,i)),img(:,:,j,i),out_size);
                end
            end
            imgvis({theta, img, Z, layer.Weights});
        end
    end
end

function  img=transform(theta, im, out_size)
one=dlarray(1);
a1=(-1:2/(out_size(1)-1):1)';
a2(1:out_size(2))=one;
a=a1*a2;
b1(1:out_size(1))=one;
b2=(-1:2/(out_size(2)-1):1);
b=(b1')*b2;
grid(1,:)=reshape(a, 1, []); 
grid(2,:)=reshape(b, 1, []); 
grid(3,:)=one;
T_g=theta*grid;
X=(T_g(1,:)+1);
Y=(T_g(2,:)+1);
x=X*(out_size(2))/2;
y=Y*(out_size(1))/2;
x0=fix(x); 
x1=x0+1; 
y0=fix(y); 
y1=y0+1;
x0=C_B_V(x0, out_size(2),1); 
x1=C_B_V(x1, out_size(2),1);
y0=C_B_V(y0, out_size(1),1); 
y1=C_B_V(y1, out_size(1),1);
base_y0=y0*out_size(2); 
base_y1=y1*out_size(2);
im_flat=(reshape(im, [], 1));
Ai1=(x1-x); 
Ai2=(y1-y);
Bi1=(y-y0); 
Ci1=(x-x0);
A1=(Ai1.*Ai2)';
B1=(Ai1.*Bi1)';
C1=(Ci1.*Ai2)';
D1=(Ci1.*Bi1)';
Ai=base_y0+x0+1;
Bi=base_y1+x0+1;
Ci=base_y0+x1+1;
Di=base_y1+x1+1;
A2=im_flat(Ai);
B2=im_flat(Bi);
C2=im_flat(Ci);
D2=im_flat(Di);
A=(A1).*A2;
B=(B1).*B2;
C=(C1).*C2;
D=(D1).*D2;
sum=(A+B+C+D);
img=reshape(sum,[out_size(1),out_size(2)]);
function x=C_B_V(x,max,min)
x(x>max-1)=max-1; 
x(x<min)=min; 
end
end

function imgvis(in)
subplot(1,2,1);
imshow(uint8(extractdata(in{2}(:,:,:,1))));
title(['Weights',{num2str((extractdata(in{4}(:,:,:,1))))}]);
subplot(1,2,2);
imshow(uint8(extractdata(in{3}(:,:,:,1))));
title(['Theta',{num2str((extractdata(in{1}(:,:,:,1))))}]);
end