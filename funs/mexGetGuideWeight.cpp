#include "mex.h"
#include <math.h>
#include <stdlib.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *Img;
    double delta, alpha;
    int rowNum, colNum, rc, nChannel, rSmooth, stride;
    
    Img = mxGetPr(prhs[0]);  // guidance image
    rSmooth = (int)mxGetScalar(prhs[1]);  // neighborhood radius in the smoothness term
    delta = mxGetScalar(prhs[2]); // delta for the guidance weight, defined in Eq. (6)
    alpha = mxGetScalar(prhs[3]);  // alpha for the guidance weight, defined in Eq. (6)
    stride = (int)mxGetScalar(prhs[4]);  // stride of the neighborhood, defined in Eq. (21)
    
    
    /************  Parameters *******************/
    rowNum = (int)mxGetDimensions(prhs[0])[0]; // row number 
    colNum = (int)mxGetDimensions(prhs[0])[1]; // column number
    rc = rowNum * colNum;
    
    if(mxGetNumberOfDimensions(prhs[0]) > 2) 
        nChannel = (int)mxGetDimensions(prhs[0])[2]; // multi-channel image
    else
        nChannel = 1; // single channel image
    
    
    /************* output memory allocate **********************/
    plhs[0] = mxCreateSparse(rc, rc, (2 * rSmooth + 1) * (2 * rSmooth + 1) * rc, mxREAL); 
    double *WeightPr = mxGetPr(plhs[0]);
    mwIndex *WeightIr = mxGetIr(plhs[0]);
    mwIndex *WeightJc = mxGetJc(plhs[0]);
    
    
    /************ compute weight *************/
    int i, j, k, s, t, counter, cenCoor, neiCoor, zero = 0, krc;
    double diff, diffSum;
    
    WeightJc[0] = 0;
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
                    
                    if((s == zero) && (t == zero)) continue; // slides to the central pixel 
                    
                    diffSum = 0;
                    neiCoor = (j + t) * rowNum + i + s; // neighbor pixel coordinate

                    for(k = 0; k < nChannel; k ++)
                    {
                        krc = k * rc;
                        diff = Img[krc + cenCoor] - Img[krc + neiCoor];
                        diffSum += fabs(diff);
                    }
                    
                    WeightPr[counter] = 1 / pow((delta + diffSum / (double)nChannel), alpha);
                    WeightIr[counter] = neiCoor;
                    counter++;
                }    
            }
            
            WeightJc[cenCoor + 1] = counter; 
        } 
    }  
}

