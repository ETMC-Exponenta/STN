clc; clear all;

%Прямое преобразование
img=dlarray(double(imresize(imread('4028965.jpg'),[200,200])));
theta = dlarray([1, 0, 1, 1, 0, 0]);
theta=reshape(theta,2,3,1,[]);
out_size=[size(img,1),size(img,2)];
for i=1:size(theta,4)
    for j=1:size(img,3)
        Z(:,:,j,i)=transform(theta(:,:,1,i),img(:,:,j,i),out_size);
    end
end
figure(1);
subplot(1,2,1);
imshow(uint8(extractdata(img(:,:,:,1))));
subplot(1,2,2);
imshow(uint8(extractdata(Z(:,:,:,1))));
title({num2str(extractdata(theta(:,:,:,1)))});

%Обратное преобразование
img=Z;
T =extractdata(theta);
T=inv(reshape(T(1:4),[2,2])); T(1,3)=0; 
theta=dlarray(T);
theta=reshape(theta,2,3,1,[]);
out_size=[size(img,1),size(img,2)];
for i=1:size(theta,4)
    for j=1:size(img,3)
        Z(:,:,j,i)=transform(theta(:,:,1,i),img(:,:,j,i),out_size);
    end
end
figure(2);
subplot(1,2,1);
imshow(uint8(extractdata(img(:,:,:,1))));
subplot(1,2,2);
imshow(uint8(extractdata(Z(:,:,:,1))));
title({num2str(extractdata(theta(:,:,:,1)))});

function  img=transform(theta, im, out_size)
grid(1,:)=reshape(((-1:2/(out_size(1)-1):1)'*ones(1,out_size(2))), 1, []); 
grid(2,:)=reshape((ones(out_size(1),1)*(-1:2/(out_size(2)-1):1)), 1, []); 
grid(3,:)=ones(size(out_size(1)*out_size(2)));
T_g=theta*grid;
x=(T_g(1,:)+1)*(out_size(2))/2;
y=(T_g(2,:)+1)*(out_size(1))/2;
x0=fix(x); x1=x0+1; 
y0=fix(y); y1=y0+1;
x0=C_B_V(x0, out_size(2)); x1=C_B_V(x1, out_size(2));
y0=C_B_V(y0, out_size(1)); y1=C_B_V(y1, out_size(1)); 
base_y0=y0*out_size(1); base_y1=y1*out_size(1);
im_flat=(reshape(im, [], 1));
A=(((x1-x).*(y1-y))').*im_flat(base_y0+x0+1);
B=(((x1-x).*(y-y0))').*im_flat(base_y1+x0+1);
C=(((x-x0).*(y1-y))').*im_flat(base_y0+x1+1);
D=(((x-x0).*(y-y0))').*im_flat(base_y1+x1+1);
img=reshape((A+B+C+D),[out_size(1),out_size(2)]);
function x=C_B_V(x,max) 
x(x>max-1)=max-1; 
x(x<1)=1; 
end
end