/**
 *  FILE:    SSA.cu
 *  AUTHOR:  Pilsung Kang
 *  CREATED: July 16, 2008
 *  LAST MODIFIED: Aug 14, 2017 
 *             BY: Pilsung Kang
 *             TO: make it work on CUDA 8 on GTX 1080ti 
 *
 *  SUMMARY:
 *
 *  NOTES: Adapted from StochKit
 *  TO DO: Making it work on CUDA 8
 *
 * cutil.h was removed:
 * - CUDA_SAFE_CALL ==> checkCudaErrors
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

// Helper functions and utilities to work with CUDA
// Ned to specify the "CUDA_Sample_dir/common/inc" dir when compiling
#include <helper_functions.h>
#include <helper_cuda.h> 

#include <curand.h>
#include <curand_kernel.h>


/* problem parameters and cuda threads launch geometry */
#include "prob_params.h"

__global__ void ssa_kernel(curandState_t* states, int *x, float *ftime);

static const int x[NX] = {1200, 600, 0};

static void init_x_array(int *xarr)
{
    for (int i=0; i<NTHREADS; i++)
        for (int j=0; j<NX; j++) 
            xarr[NX*i+j] = x[j];
}

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {
    /* we have to initialize the state */
    curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
                blockIdx.x*blockDim.x+threadIdx.x, /* the sequence number should be different for each core (unless you want all
                           cores to get the same sequence of numbers for some reason - use thread id! */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &states[blockIdx.x*blockDim.x+threadIdx.x]);
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a
 * random float between 0.0 and 1.0 into each */
__global__ void randoms(curandState_t* states, float* numbers) {
    /* curand works like rand - except that it takes a state as a parameter */
	numbers[blockIdx.x*blockDim.x+threadIdx.x] = curand_uniform(&states[blockIdx.x*blockDim.x+threadIdx.x]);
}


int main(int argc, char** argv) {

    // In my Linux config with 1080ti, the id is 0
    int devID = 0;

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    curandState_t* states;

    /* allocate space on the GPU for the random states */
    cudaMalloc((void**) &states, NTHREADS * sizeof(curandState_t));

    dim3 dimBlock(XBLOCKSIZE, YBLOCKSIZE);
    dim3 dimGrid(XGRIDSIZE, YGRIDSIZE);

    /* invoke the GPU to initialize all of the random states */
    init<<<dimGrid, dimBlock>>>(time(0), states);

	/* allocate an array of floats on the CPU and GPU */
	float cpu_nums[NTHREADS];
	float* gpu_nums;
	cudaMalloc((void**) &gpu_nums, NTHREADS * sizeof(float));

	/* invoke the kernel to get some random numbers */
	randoms<<<dimGrid, dimBlock>>>(states, gpu_nums);

	/* copy the random numbers back */
	cudaMemcpy(cpu_nums, gpu_nums, NTHREADS * sizeof(float), cudaMemcpyDeviceToHost);

	/* print them out */
	// for (int i = 0; i < NTHREADS; i++) {
	// 	printf("%10.10f\n", cpu_nums[i]);
	// }


    // allocate dev mem for x (specifes) and copy initial values
    int *x_array = (int *) malloc(NX*NTHREADS*sizeof(int));
    init_x_array(x_array);
    int *dev_x_array;

    checkCudaErrors(cudaMalloc((void **)&dev_x_array, sizeof(int)*NX*NTHREADS));
    checkCudaErrors(cudaMemcpy(dev_x_array, x_array, sizeof(int)*NX*NTHREADS, cudaMemcpyHostToDevice));

    // allocate dev mem for final time and copy initial values
    float *finalT_array = (float *) malloc(NTHREADS*sizeof(float));
    float *dev_finalT_array;

    checkCudaErrors(cudaMalloc((void **)&dev_finalT_array, sizeof(float)*NTHREADS));

    // Referenced from the matrixMul CUDA sample
    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Record the start event
    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    printf("Number of CUDA threads = %u \n", NTHREADS);

    // Execute the SSA kernel
    ssa_kernel<<<dimGrid, dimBlock>>>(states, dev_x_array, dev_finalT_array);
    // device sync might be unnecessary...
	cudaDeviceSynchronize();  

    // Record the stop event
    error = cudaEventRecord(stop, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    printf("Kernel exec time = %.3f msec\n", msecTotal);

    // copy result from device to host
    checkCudaErrors(cudaMemcpy(x_array, dev_x_array, sizeof(int)*NX*NTHREADS, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(finalT_array, dev_finalT_array, sizeof(float)*NTHREADS, cudaMemcpyDeviceToHost));

    // for (int i=0; i<NTHREADS; i++) {
    //     printf("Tid %d at final time %f: %d\t\t%d\t\t%d\n",
    //         i, finalT_array[i], x_array[i*NX], x_array[i*NX+1], x_array[i*NX+2]);
    // }

	/* free the memory we allocated for the states and numbers */
    free(x_array); 
    free(finalT_array);
	checkCudaErrors(cudaFree(states));
	checkCudaErrors(cudaFree(gpu_nums));
    checkCudaErrors(cudaFree(dev_x_array)); 
    checkCudaErrors(cudaFree(dev_finalT_array));

    return 0;
}

