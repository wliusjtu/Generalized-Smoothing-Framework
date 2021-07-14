clear; close all;
addpath('./funs')

%% load image
Img =imread('./imgs/texture_removal/01.jpg');


%% parameters (SP-1 mode)
smallNum = 1e-3;
thr = 1; 

lambda = 0.5;  % 0.5 for "01", 1.25 for "02", 0.75 for "03", 0.7 for "04"
rData = 1;
rSmooth = 1;
aData = smallNum;
aSmooth = smallNum;
bData = thr; 
bSmooth = thr;
alpha = 0.5;
stride = 1;
iterNum = 10;


%% smooth and show results
Out = generalized_smooth(Img, Img, lambda, rData, rSmooth, aData, bData, aSmooth, bSmooth, alpha, stride, iterNum);

figure; imshow(uint8([Img, Out]));