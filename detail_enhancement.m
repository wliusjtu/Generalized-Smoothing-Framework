clear; close all;
addpath('./funs')

%% images
Img = imread('./imgs/detail_enhancement/tree.png');

%% parameters (SP-2 mode)
smallNum = 1e-3;  % try 1e-4 for different smoothing results
thr = 1;  % this value should be larger than the maximum intensity value

lambda = 20;
rData = 1;
rSmooth = 1;
aData = smallNum;
aSmooth = smallNum;
bData = thr; 
bSmooth = thr;
alpha = 0.2;
stride = 1;
iterNum = 1;


%% smooth
Out_SP = generalized_smooth(Img, Img, lambda, rData, rSmooth, aData, bData, aSmooth, bSmooth, alpha, stride, iterNum);


%% parameters (EP-1 mode/WLS filter)
lambda = 1;
rData = 0;
% the rest remain the same

%% smooth
Out_EP = generalized_smooth(Img, Img, lambda, rData, rSmooth, aData, bData, aSmooth, bSmooth, alpha, stride, iterNum);


%% parameters (EP-1 mode/WLS filter, another way to achieve WLS smoothing)
% lambda = 1;
% rData = 0;
% aSmooth = thr;
% alpha = 1.2;
% % the rest remain the same
% 
% %% smooth
% Out_EP = generalized_smooth(Img, Img, lambda, rData, rSmooth, aData, bData, aSmooth, bSmooth, alpha, stride, iterNum);


%% show and compare the results
Img = double(Img);
figure; imshow(uint8(Img))
figure; imshow(uint8([Out_SP, Out_EP]))
figure; imshow(uint8([Img + 5 * (Img - Out_SP), Img + 5 * (Img - Out_EP)]))

