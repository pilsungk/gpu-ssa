//*****************************************************************************|
//*  FILE:    ssa_kernel.cu
//*
//*  AUTHOR:  Pilsung Kang
//*
//*  CREATED: July 19, 2008
//*
//*  LAST MODIFIED:  
//*             BY:  
//*             TO:  
//*
//*  SUMMARY:
//*
//*
//*  NOTES: 
//*
//*
//*  TO DO:
//*
//*
//*****************************************************************************|

#include <curand.h>
#include <curand_kernel.h>

/* problem parameters and cuda threads launch geometry */
#include "prob_params.h"

__global__ void ssa_kernel(curandState_t* states, int *x, float *ftime)
{

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int tid = bx*blockDim.x + tx;
    const int xBegin = NX * tid;
    const int xSharedBegin = tx*NX;

    __shared__ int xShared[NX*XBLOCKSIZE];
    __shared__ int nu[DIMX_NU][DIMY_NU];
    __shared__ float proprates[NCHANNEL];
    __shared__ int done[XBLOCKSIZE];


    if (tx == 0) {
        nu[0][0] = -1; nu[0][1] = 1;  nu[0][2] = 0;
        nu[1][0] = 1;  nu[1][1] = -1; nu[1][2] = -1;
        nu[2][0] = 0;  nu[2][1] = 0;  nu[2][2] = 1;

        proprates[0] = 1.0f;
        proprates[1] = 2.0f;
        proprates[2] = 0.00005f;
    }

    for (int i=0; i<NX; i++) {
        xShared[xSharedBegin+i] = x[xBegin+i];
    }

    float curTime = 0.0;
    
    float a0, a[3];
    float f, jsum, tau;
    int rxn;
    float rand1, rand2;
    int total_done;
    int counter = 0;

    done[tx] = 0;

    rand1 = curand_uniform(&states[tid]);
    rand2 = curand_uniform(&states[tid]);

    while (1) {
        counter++;

        if (!done[tx]) {
            // take step -- 1. choose the channel to fire
            //printf("x0 = %d, x1 = %d, x2 = %d\n", x[xBegin], x[xBegin+1], x[xBegin+2]);
            a[0] = xShared[xSharedBegin]*proprates[0];
            a[1] = xShared[xSharedBegin+1]*proprates[1];
            a[2] = xShared[xSharedBegin+1]*proprates[2];

            a0 = a[0] + a[1] + a[2];
            f = rand1 * a0;

            jsum = 0.0;

            for(rxn=0; jsum < f; rxn++) jsum += a[rxn];
            rxn--;


            // take step -- 2. fire the chosen channel
            for (int i=0; i<NX; i++) {
                xShared[xSharedBegin+i] += nu[i][rxn];
            }

            // take step -- 3. calculate the time step
            tau = -logf(rand2) / a0;
            curTime += tau;

            // negative state check
            for (int i=0; i<NX; i++) {
                if (xShared[xSharedBegin+i] < 0) {
                    for (int j=0; j<NX; j++) { 
                        xShared[xSharedBegin+j] -= nu[j][rxn];
                    }

                    curTime -= tau;
                    break;
                }
            }

            if (curTime > FINALTIME) done[tx] = 1;
        }

        __syncthreads();
        
        if (!done[tx]) total_done = 0;
        else {
            total_done = 1;
            for (int i=0; i<blockDim.x; i++) total_done *= done[i];
        }

        __syncthreads();

        if (total_done) break;

        rand1 = curand_uniform(&states[tid]);
        rand2 = curand_uniform(&states[tid]);
    }

    ftime[tid] = curTime;

    for (int i=0; i<NX; i++) {
        x[xBegin+i] = xShared[xSharedBegin+i];
    }
}

