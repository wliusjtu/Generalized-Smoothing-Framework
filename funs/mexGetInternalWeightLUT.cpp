#include "mex.h"
#include <math.h>
#include <stdlib.h>

void GetGaussianSpatialKernel(double *WeightSpatial, int r, double sigmaS);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *ImgOri, *ImgNew;
    double sigmaSData, sigmaSSmooth, *WeightLUTData, *WeightLUTSmooth, aData, aSmooth, bData, bSmooth;
    int rowNum, colNum, rc, nChannel, rData, rSmooth, winSizeData, winSizeSmooth, stride;
    
    
    /**********  Inputs *******************/
    ImgOri = mxGetPr(prhs[0]); // the input image, "f" in eq. (4)
    ImgNew = mxGetPr(prhs[1]); // the smoothed image in the current iteration, "u^k" in Eq. (15s)
    rData = (int)mxGetScalar(prhs[2]); // neighborhood radius of the data term, defined in Eq. (4)
    rSmooth = (int)mxGetScalar(prhs[3]); // neighborhood radius of the smoothness term, defined in Eq. (4)
    aData = mxGetScalar(prhs[4]); // parameter "a" in the truncated Huber penalty of the data term, defined in Eq. (1)
    aSmooth = mxGetScalar(prhs[5]); // parameter "a" in the truncated Huber penalty of the smoothness term, defined in Eq. (1)
    bData = mxGetScalar(prhs[6]); // parameter "b" in the truncated Huber penalty of the data term, defined in Eq. (1)
    bSmooth = mxGetScalar(prhs[7]); // parameter "b" in the truncated Huber penalty of the smoothness term, defined in Eq. (1)
    stride = (int)mxGetScalar(prhs[8]);  // neighborhood stride, defined in Eq. (21)
    
    sigmaSData = (double) rData; // spatial sigma for the data term, defined in Eq. (5)
    sigmaSSmooth = (double) rSmooth; // spatial sigma for the smooth term, defined in Eq. (5)
    
    
    /************  Parameters *******************/
    rowNum = (int)mxGetDimensions(prhs[0])[0]; // row number 
    colNum = (int)mxGetDimensions(prhs[0])[1]; // column number
    rc = rowNum * colNum;
    
    if(mxGetNumberOfDimensions(prhs[0]) > 2) 
        nChannel = (int)mxGetDimensions(prhs[0])[2]; // multi channel image
    else
        nChannel = 1; // single channel image
    
    
    /**********************gaussian spatial kernel***********************/
    winSizeData = 2 * rData + 1;
    winSizeSmooth = 2 * rSmooth + 1;
    
    double *WeightSpatialData = mxGetPr(mxCreateDoubleMatrix(winSizeData, winSizeData, mxREAL));// spatial weight for the data term
    double *WeightSpatialSmooth = mxGetPr(mxCreateDoubleMatrix(winSizeSmooth, winSizeSmooth, mxREAL));// spatial weight for the smoothness term
    
    GetGaussianSpatialKernel(WeightSpatialData, rData, sigmaSData);
    GetGaussianSpatialKernel(WeightSpatialSmooth, rSmooth, sigmaSSmooth);
    
    
    //****************compute LUT*********************/
    int maxRange = nChannel * (255 + 50) * (255 + 50);  // we quantize the intensity values of the guidance image into 65025 bins, 
                                                                                  // but their values shoud meet |I_max - I_min|<=1
    WeightLUTData = mxGetPr(mxCreateDoubleMatrix(1, maxRange, mxREAL)); 
    WeightLUTSmooth = mxGetPr(mxCreateDoubleMatrix(1, maxRange, mxREAL)); 
        
    for(int i = 0; i < maxRange; i ++)
    {
        if(i <= (int)(aData * 65025.0 * nChannel))  WeightLUTData[i] = 0.5 / aData;
        else if(i > (int)(bData * 65025.0 * nChannel))  WeightLUTData[i] = 1e-7;
        else WeightLUTData[i] = 0.5 /((double) i / (65025.0 * (double)nChannel));
        
        if(i <= (int)(aSmooth * 65025.0 * nChannel))  WeightLUTSmooth[i] = 0.5 / aSmooth;
        else if(i > (int)(bSmooth * 65025.0 * nChannel))  WeightLUTSmooth[i] = 1e-7;
        else WeightLUTSmooth[i] = 0.5 / ((double) i / (65025.0 * (double)nChannel));
    }
    
    /************* Output memory allocate **********************/
    plhs[0] = mxCreateSparse(rc, rc, (2 * rData + 1) * (2 * rData + 1) * rc, mxREAL); 
    double *WeightDataPr = mxGetPr(plhs[0]);
    mwIndex *WeightDataIr = mxGetIr(plhs[0]);
    mwIndex *WeightDataJc = mxGetJc(plhs[0]);
    
    plhs[1] = mxCreateSparse(rc, rc, (2 * rSmooth + 1) * (2 * rSmooth + 1) * rc, mxREAL); 
    double *WeightSmoothPr = mxGetPr(plhs[1]);
    mwIndex *WeightSmoothIr = mxGetIr(plhs[1]);
    mwIndex *WeightSmoothJc = mxGetJc(plhs[1]);
    
    
    /************ compute weight *************/
    int i, j, k, s, t, counter, cenCoor, neiCoor, zero = 0, krc;
    double diff, diffSum;
    
    
    /************** Data term *********************/
    WeightDataJc[0] = 0;
    counter = 0;
    
    for(j = 0; j < colNum; j ++)
    {        
        for(i = 0; i < rowNum; i ++)
        {
            cenCoor = j * rowNum + i; // central pixel coordinate
            
            for(t = - rData; t <= rData; t++)
            {
                if(((j + t) < 0) || ((j + t) > colNum - 1)) continue; // boundary
                 
                for(s = - rData; s <= rData; s++)
                {
                    if(((i + s) < 0) || ((i + s) > rowNum - 1)) continue; // boudary
                    if((s == zero) && (t == zero)) continue;// slides to the central pixel 
                    
                    diffSum = 0;
                    neiCoor = (j + t) * rowNum + i + s; // neighbor pixel coordinate

                    for(k = 0; k < nChannel; k ++)
                    {
                        krc = k * rc;
                        diff = ImgNew[krc + cenCoor] - ImgOri[krc + neiCoor];
                        diffSum += fabs(diff);
                    }
                   

                    WeightDataPr[counter] = WeightLUTData[(int)(diffSum * 65025.0)] * WeightSpatialData[(t + rData) * winSizeData + s + rData];
                    WeightDataIr[counter] = neiCoor;       
                    counter++;
                }
            }
            
            WeightDataJc[cenCoor + 1] = counter;
        } 
    }  
    
    
    /************** Smoothness term *********************/
    
    WeightSmoothJc[0] = 0;
    counter = 0;
    
    for(j = 0; j < colNum; j ++)
    {        
        for(i = 0; i < rowNum; i ++)
        {
            cenCoor = j * rowNum + i; // central pixel coordinate
            
            for(t = - rSmooth; t <= rSmooth; t += stride)
            {
                if(((j + t) < 0) || ((j + t) > colNum - 1)) continue; // boundary
                 
                for(s = - rSmooth; s <= rSmooth; s += stride)
                {
                    if(((i + s) < 0) || ((i + s) > rowNum - 1)) continue; // boudary
                    if((s == zero) && (t == zero)) continue;// slides to the central pixel 
                    
                    diffSum = 0;
                    neiCoor = (j + t) * rowNum + i + s; // neighbor pixel coordinate

                    for(k = 0; k < nChannel; k ++)
                    {
                        krc = k * rc;
                        diff = ImgNew[krc + cenCoor] - ImgNew[krc + neiCoor];
                        diffSum += fabs(diff);
                    }

                    WeightSmoothPr[counter] = WeightLUTSmooth[(int)(diffSum * 65025.0)] * WeightSpatialSmooth[(t + rSmooth) * winSizeSmooth + s + rSmooth];
                    WeightSmoothIr[counter] = neiCoor;                    
                    counter++;   
                }        
            }
            
            WeightSmoothJc[cenCoor + 1] = counter; 
        }   
    }  
}


/***************** Function **********************/
void GetGaussianSpatialKernel(double *WeightSpatial, int r, double sigmaS)
{
    int i, j, winSize;
    double WeightSum = 0;
    
    winSize = 2 * r + 1;
    
    for(i = 0; i < winSize; i ++)
    {
        for(j = 0; j < winSize; j ++)
        {
            WeightSpatial[j * winSize + i] = exp(- (double)((i - r) * (i - r) + (j - r) * (j - r)) / (2.0 * sigmaS * sigmaS));
            WeightSum += WeightSpatial[j * winSize + i];
        }
    }
    
    for(i = 0; i < winSize; i ++)
        for(j = 0; j < winSize; j ++)
            WeightSpatial[j * winSize + i] /= WeightSum;
    
}

