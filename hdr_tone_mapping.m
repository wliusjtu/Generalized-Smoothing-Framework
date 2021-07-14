% this code is based on the HDR tone mapping framework developed by Zeev Farbman et al. Their code can be downloed at: https://www.cs.huji.ac.il/~danix/epd/

clear; close all
addpath('./funs')

%% load the hdr image
hdr = double(hdrread('./imgs/hdr_tone_mapping/doll.hdr'));
I = 0.2989*hdr(:,:,1) + 0.587*hdr(:,:,2) + 0.114*hdr(:,:,3);
logI = log(I+eps);


%% parameters (SP-2 mode)
smallNum = 1e-3;  % try 1e-4 for different results
thr = 1; % this value should be larger than the maximum intensity value

lambda = 200;
rData = 1;
rSmooth = 1;
aData = smallNum;
aSmooth = smallNum;
bData = thr; 
bSmooth = thr;
alpha = 0.2;
stride = 1;
iterNum = 1;

base = generalized_smooth(logI, logI, lambda, rData, rSmooth, aData, bData, aSmooth, bSmooth, alpha, stride, iterNum);

% Compress the base layer and restore detail
compression = 0.25;  % this parameter should be adjusted for different inputs
detail = logI - base;
OUT = base * compression +  1.2 * detail;
OUT = exp(OUT);

% Restore color
OUT = OUT./I;
OUT = hdr .* padarray(OUT, [0 0 2], 'circular' , 'post');


% Finally, shift, scale, and gamma correct the result
gamma = 1.0/2.2;
bias = -min(OUT(:));
gain = 0.45;  % this parameter should be adjusted for different inputs
OUT_SP = (gain * (OUT + bias)).^gamma;


%% parameters (EP-1 mode/ WLS filter)
lambda = 2.5;
rData = 0;
% the rest remain the same

base = generalized_smooth(logI, logI, lambda, rData, rSmooth, aData, bData, aSmooth, bSmooth, alpha, stride, iterNum);

% Compress the base layer and restore detail
compression = 0.25;
detail = logI - base;
OUT = base * compression +  1.2 * detail;
OUT = exp(OUT);

% Restore color
OUT = OUT./I;
OUT = hdr .* padarray(OUT, [0 0 2], 'circular' , 'post');


% Finally, shift, scale, and gamma correct the result
gamma = 1.0/2.2;
bias = -min(OUT(:));
gain = 0.45;
OUT_EP = (gain * (OUT + bias)).^gamma;


%% show and compare the results
figure; imshow([OUT_SP, OUT_EP])
    