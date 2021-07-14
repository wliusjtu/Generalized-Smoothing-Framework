clear; close all;
addpath('./funs')

name = 'art';

%% images
Img = imread(['./imgs/guided_depth_map_upsampling/', name, '/depth_3_n.png']);
ImgGuide = imread(['./imgs/guided_depth_map_upsampling/', name, '/', name, '_color.png']);

[m, n, ~] = size(ImgGuide);
Img =imresize(Img, [m, n]);


%% parameters (EP & SP mode)
smallNum = 1e-3;
thr = 0.08;  % 0.1/0.1/0.08/0.07 for 2x/4x/8x/16x upsampling

rData = 5;
rSmooth = 5;
aData = smallNum;
aSmooth = smallNum;
bData = thr; 
bSmooth = thr;
alpha = 0.5;
iterNum = 10;


%% stride=1
lambda = 0.5; % 0.1/0.25/0.5/0.95 for 2x/4x/8x/16x upsampling
stride = 1;

Out_s1 = generalized_smooth(Img, ImgGuide, lambda, rData, rSmooth, aData, bData, aSmooth, bSmooth, alpha, stride, iterNum);


%% stride=2
lambda = 1.6; % 0.35/0.75/1.6/3.0 for 2x/4x/8x/16x upsampling
stride = 2;
% the rest parameters remain the same

Out_s2 = generalized_smooth(Img, ImgGuide, lambda, rData, rSmooth, aData, bData, aSmooth, bSmooth, alpha, stride, iterNum);


%% compute MAE and show results
ImgOri = double(imread(['./imgs/guided_depth_map_upsampling/', name, '/', name, '_big.png']));

MAE_s1 = mean(mean(abs(Out_s1 - ImgOri)));
MAE_s2 = mean(mean(abs(Out_s2 - ImgOri)));
fprintf('MEA for stride=1 is %f, MAE for stride=2 is %f\n', MAE_s1, MAE_s2);

figure; imshow(uint8([ImgOri, Out_s1, Out_s2]));
