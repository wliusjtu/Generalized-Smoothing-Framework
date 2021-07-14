clear; close all;
addpath('./funs')

%% images
Img = im2double(imread('./imgs/clip_art_compression_removal/01.jpg'));


%% parameters (EP & SP mode)
smallNum = 1e-3;
thr = 0.075; % 0.075 for "01", 0.15 for "02" and "03"

lambda = 0.3;  % 0.3 for "01", 0.4 for "02" and "03"
rData = 2;
rSmooth = 2;
aData = smallNum;
aSmooth = smallNum;
bData = thr; 
bSmooth = thr;
alpha = 0.5;
stride = 1;
iterNum = 10;


%% smooth and show results
Out = generalized_smooth(Img, Img, lambda, rData, rSmooth, aData, bData, aSmooth, bSmooth, alpha, stride, iterNum);

figure; imshow(Out);