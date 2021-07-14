clear; close all;
addpath('./funs')

%% images
Img = imread('./imgs/flash_no_flash_filtering/cave_noflash.png');
ImgGuide = imread('./imgs/flash_no_flash_filtering/cave_flash.png');


%% parameters (EP & SP mode)
smallNum = 1e-3;
thr = 0.15; % this value should be adjusted according to different noise levels

lambda = 0.1;
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
Out = generalized_smooth(Img, ImgGuide, lambda, rData, rSmooth, aData, bData, aSmooth, bSmooth, alpha, stride, iterNum);

figure; imshow(uint8([Img, Out]));