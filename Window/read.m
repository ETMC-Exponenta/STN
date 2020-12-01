function imgData = read(way)
T=rand*10*randi([-1,1],1,1);
tform = maketform('affine', [cosd(T) -sind(T) 0; sind(T) cosd(T) 0; 0 0 1]);
img=(imresize(double(imread(way)),[50,50]));
imgData =dlarray(imtransform(img,tform,'size',size(img),'fill',255));
imgData = C_B_V(fix(imgData+(rand*randi([-50,50],1,1))));
imgData(imgData==(max(max(max(imgData)))))=255;
end
function x=C_B_V(x)
x(x>255)=255; 
x(x<0)=0; 
end