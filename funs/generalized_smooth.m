%   Distribution code Version 1.0 -- 12/07/2021 by Wei Liu Copyright 2021
%
%   The code is created based on the method described in the following papers
%   [1] "A generalized framework for edge-preserving and structure-preserving image smoothing",
%        Wei Liu, Pingping Zhang, Yinjie Lei, Xiaolin Huang, Jie Yang and Ian Reid, AAAI, 2020.
%   [2] "A generalized framework for edge-preserving and structure-preserving image smoothing",
%        Wei Liu, Pingping Zhang, Yinjie Lei, Xiaolin Huang, Jie Yang and Michael Ng, 
%        IEEE Transactions on Pattern Anaysis and Machine Intelligence, 2021.
%  
%   The code and the algorithm are for non-comercial use only.


%  ---------------------- Input------------------------
%  Img:                input image, can be gray image or RGB color image
%  ImgGuide:       guide image, can be gray image or RGB color image
%  lambda:           lambda in Eq. (4), control smoothing strength
%  rData:             neighborhood radius of the data term in Eq. (4)
%  rSmooth:         neighborhood radius of the smoothness term in Eq. (4)
%  aData:             "a" of the truncated Huber penalty in the data term 
%  aSmooth:         "a" of the truncated Huber penalty in the smoothness term 
%  bData:             "b" of the truncated Huber penalty in the data term 
%  bSmooth:         "b" of the truncated Huber penalty in the smoothness term 
%  alpha:              "alpha" of the guidance weight in Eq. (6)
%  stride:              stride of the neighborhood, defined in Eq. (21)


%  ---------------------- Output------------------------
%  Out:             smoothed image


% this function perform normalization of the input images such that their
% maximum values I_max and minimum values I_min meet |I_max - I_min|<=1,
% thus setting bData>=1 and bSmooth >=1 will result in no trunctation in
% the data term and the smoothness term.
% the output are normalized back to the range of the input.


function Out = generalized_smooth(Img, ImgGuide, lambda, rData, rSmooth, aData, bData, aSmooth, bSmooth, alpha, stride, iterNum)

fprintf('\n*************************** Generalized Smoothing Framework ***************************\n');

% the input images must be in double data type
Img = double(Img);
ImgGuide = double(ImgGuide);

% normalization
I_max_img = max(Img(:));
I_min_img = min(Img(:));
Img = Img ./ (I_max_img - I_min_img);  

I_max_guide = max(ImgGuide(:));
I_min_guide = min(ImgGuide(:));
ImgGuide = ImgGuide ./ (I_max_guide - I_min_guide);

% get image size
[row, col, cha] = size(Img);
rc = row * col;

% measure time
tTotal = tic;

% compute guidance weights 
delta = 1e-3;  % delta in Eq. (6)
tStart = tic;
GuideWeight = mexGetGuideWeight(ImgGuide, rSmooth, delta, alpha, stride);
tEnd = toc(tStart);
fprintf('Computing guidance weight costs %f seconds\n', tEnd);

Out = Img;  % initialize the smoothed output with the input image
X1 = reshape(Img, [rc, cha]);

for k = 1: iterNum
    fprintf("\niteration #%d\n", k);
    
    tStart = tic;
    [InternalWeightData, InternalWeightSmooth] = mexGetInternalWeight(Img, Out, rData, rSmooth, aData, aSmooth, bData, bSmooth, stride);
    tEnd = toc(tStart);
    fprintf('Computing internal weight costs %f seconds\n', tEnd);
    
    tStart = tic;
    if rData == 0
        X0 = reshape(Img, [rc, cha]);
        L = speye(rc);
    else
        X0 = InternalWeightData' * reshape(Img, [rc, cha]);
        L = diag(sum(InternalWeightData));
    end
    clear InternalWeightData
    
    CombinedWeight = GuideWeight .* InternalWeightSmooth;
    clear InternalWeightSmooth
    
    L = L + 2 * lambda * diag(sum(CombinedWeight)) - 2 * lambda * CombinedWeight;
    clear CombinedWeight
    
    tEnd = toc(tStart);
    fprintf('Computing combined weight and coefficient matrix costs %f seconds\n', tEnd);
    
    tStart = tic;
    R = ichol(L, struct('type', 'ict', 'droptol', 1e-03, 'michol','on'));
    tEnd = toc(tStart);
    fprintf('Computing preconditioner costs %f seconds\n', tEnd);

    tStart = tic;
    for c = 1: cha
        X1(:, c) = pcg(L, X0(:, c), 1e-6, 100, R, R');
    end
    tEnd = toc(tStart);
    fprintf('Solving PCG costs %f seconds\n', tEnd);
    
    clear L R

    Out = reshape(X1, [row, col, cha]);
    
end

% normalize back to the input range
Out = Out * (I_max_img - I_min_img);

tTotal = toc(tTotal);
fprintf('\n%d iterations totally cost %f seconds\n\n', iterNum, tTotal);

