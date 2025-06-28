#include <cooperative_groups.h>
#include <math.h>
#include <iostream>

#include "../include/kernel_struct.h"

namespace cg = cooperative_groups;


__device__ float cotf(float x)
{  
    return 1 / tanf(x);
};


__device__ float cscf(float x)
{
    return  1 / sinf(x);
};


static __global__ void setRnStates(curandState *rndStates, unsigned long long seed)
{
    int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    curand_init(seed+idx, idx, 0, &rndStates[idx]);
}


void initializeRandomStates(KernelParams kernelParams)
{
    cudaMemset(
        kernelParams.rngStates, 0, 
        kernelParams.numBlocks * kernelParams.blockSize * sizeof(curandState));

    setRnStates<<<kernelParams.numBlocks, kernelParams.blockSize>>>(
        kernelParams.rngStates, time(NULL));
};


__device__ void getNNIndices(int spinIdx, int numBodies, int sBs, int nnInds[4])
{
    int spinBlock = spinIdx / numBodies;
    spinIdx = spinIdx % numBodies;

    *(nnInds) = sBs * (spinIdx/sBs) + (spinIdx + 1) % sBs + spinBlock * (sBs * sBs);
    *(nnInds + 1) = sBs * (spinIdx/sBs) + (spinIdx - 1 + sBs) % sBs + spinBlock * (sBs * sBs);
    *(nnInds + 2) = (spinIdx + sBs) % (sBs * sBs) + spinBlock * (sBs * sBs);
    *(nnInds + 3) = (spinIdx - sBs + (sBs * sBs)) % (sBs * sBs) + spinBlock * (sBs * sBs);
};

    
__device__ float3 computeNBIntearaction(float2* buffer, int idx, int numTiles, cg::thread_block cta) 
{
  extern __shared__ float2 sharedBuffer[];
  float3 interaction = {0.0f, 0.0f, 0.0f, };

  for (int tile = 0; tile < numTiles; tile++) 
  {
        sharedBuffer[threadIdx.x] = buffer[tile * blockDim.x + threadIdx.x];
        cg::sync(cta);
        #pragma unroll 128
        for (unsigned int counter = 0; counter < blockDim.x; counter++) 
        {
            interaction.x += sinf(sharedBuffer[counter].x) * cosf(sharedBuffer[counter].y);
            interaction.y += sinf(sharedBuffer[counter].x) * sinf(sharedBuffer[counter].y);
            interaction.z += cosf(sharedBuffer[counter].x);
        };
        cg::sync(cta);
  }
  return interaction;
}


__device__ float3 compute4BInteraction(float2* spinsBuff, 
    int spinIdx, int sBs, int numBodies, cg::thread_block cta)
{
    float3 interaction = {0.0f, 0.0f, 0.0f};
    static int nnInds[4];

    getNNIndices(spinIdx, numBodies, sBs, nnInds);

    cg::sync(cta);
    #pragma unroll 4
    for (unsigned int counter = 0; counter < 4; counter++) 
    {
        interaction.x += sinf(spinsBuff[*(nnInds + counter)].x) * cosf(spinsBuff[*(nnInds + counter)].y);
        interaction.y += sinf(spinsBuff[*(nnInds + counter)].x) * sinf(spinsBuff[*(nnInds + counter)].y);
        interaction.z += cosf(spinsBuff[*(nnInds + counter)].x);
    };
    cg::sync(cta);
    return interaction;
}


__device__ float3 compute4BInteraction(float3* spinsBuff, 
    int spinIdx, int sBs, int numBodies, cg::thread_block cta)
{
    float3 interaction = {0.0f, 0.0f, 0.0f};
    static int nnInds[4];

    getNNIndices(spinIdx, numBodies, sBs, nnInds);

    cg::sync(cta);
    #pragma unroll 4
    for (unsigned int counter = 0; counter < 4; counter++) 
    {
        interaction.x += spinsBuff[*(nnInds + counter)].x;
        interaction.y += spinsBuff[*(nnInds + counter)].y;
        interaction.z += spinsBuff[*(nnInds + counter)].z;
    };
    cg::sync(cta);
    return interaction;
}


__global__ void simulate_xy_model(float2 *newBuffQuantity, float2 *oldBuffQuantity,
    PhysicalParams params, int numTiles, curandState *rngStates)
{
    cg::thread_block cta = cg::this_thread_block();

    int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    float2 bufferQuant = oldBuffQuantity[idx];
    float sqr3 = sqrtf(3);

    float3 interaction = compute4BInteraction(
        oldBuffQuantity, idx, params.subSystemSize, params.numBodies, cta);
    curandState localState = rngStates[idx];

    bufferQuant.x += (- 2 * params.Jx * sinf(oldBuffQuantity[idx].y) * sqr3 * interaction.x + \
                    2* params.Jy * cosf(oldBuffQuantity[idx].y) * sqr3 * interaction.y ) * params.dt +  \
                    2 * params.Gamma * (cotf(oldBuffQuantity[idx].x) - cscf(oldBuffQuantity[idx].x)/sqr3) \
                    * params.dt;

    bufferQuant.y += (- 2 * params.Jx * cotf(oldBuffQuantity[idx].x) * cosf(oldBuffQuantity[idx].y) * sqr3 * interaction.x - \
                    2 * params.Jy * cotf(oldBuffQuantity[idx].x) * sinf(oldBuffQuantity[idx].y) \
                    * sqr3 * interaction.y + 2 * params.Jz * sqr3 * interaction.z) * params.dt + sqrtf(2) \
                    * sqrtf(params.Gamma) * sqrtf(1 + 2*cotf(oldBuffQuantity[idx].x)*cotf(oldBuffQuantity[idx].x) -\
                    2*cotf(oldBuffQuantity[idx].x)*cscf(oldBuffQuantity[idx].x)/sqr3) * sqrtf(params.dt) * curand_normal(&localState);

    rngStates[idx] = localState;
    newBuffQuantity[idx] = bufferQuant;
};


 __global__ void simulate(float2 *newBuffQuantity, float2 *oldBuffQuantity,
                        PhysicalParams params, int numTiles, 
                        curandState *rngStates)
{
    cg::thread_block cta = cg::this_thread_block();
    
    int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    float2 bufferQuant = oldBuffQuantity[idx];
    
    float3 interaction = computeNBIntearaction(oldBuffQuantity, idx, numTiles, cta);
    float sqr3 = sqrtf(3);

    curandState localState = rngStates[idx];

    bufferQuant.x += (- 2 * params.Jx * sinf(oldBuffQuantity[idx].y) * sqr3 * interaction.x + \
    2* params.Jy * cosf(oldBuffQuantity[idx].y) * sqr3 * interaction.y ) * params.dt +  \
    2 * params.Gamma * (cotf(oldBuffQuantity[idx].x) - cscf(oldBuffQuantity[idx].x)/sqr3) \
    * params.dt;

    bufferQuant.y += (- 2 * params.Jx * cotf(oldBuffQuantity[idx].x) * cosf(oldBuffQuantity[idx].y) * sqr3 * interaction.x - \
    2 * params.Jy * cotf(oldBuffQuantity[idx].x) * sinf(oldBuffQuantity[idx].y) \
    * sqr3 * interaction.y + 2 * params.Jz * sqr3 * interaction.z) * params.dt + sqrtf(2) \
    * sqrtf(params.Gamma) * sqrtf(1 + 2*cotf(oldBuffQuantity[idx].x)*cotf(oldBuffQuantity[idx].x) -\
     2*cotf(oldBuffQuantity[idx].x)*cscf(oldBuffQuantity[idx].x)/sqr3) * sqrtf(params.dt) * curand_normal(&localState);

    rngStates[idx] = localState;
    newBuffQuantity[idx] = bufferQuant;
};


__global__ void simulate_xyz_model(float3 *newS, float3 *oldS,
    PhysicalParams params, int numTiles, curandState *rngStates)
{
    cg::thread_block cta = cg::this_thread_block();

    int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    float3 bufferQuant = oldS[idx];
    float sqrtGamma = sqrtf(0.5 * params.Gamma);

    curandState localState = rngStates[idx];
    float rndA = curand_normal(&localState);
    float rndB = curand_normal(&localState);

    float3 interaction = compute4BInteraction(
        oldS, idx, params.subSystemSize, params.numBodies, cta);

    bufferQuant.x += ( -0.5 * params.Gamma * oldS[idx].x + 2*params.Jy * oldS[idx].z * interaction.y  - 2*params.Jz * oldS[idx].y * interaction.z) * params.dt + \
                     sqrtGamma * (1 + oldS[idx].z - oldS[idx].x * oldS[idx].x) * rndA + sqrtGamma * oldS[idx].x * oldS[idx].y * rndB;

    bufferQuant.y += ( -0.5 * params.Gamma * oldS[idx].y + 2*params.Jz * oldS[idx].x * interaction.z  - 2*params.Jx * oldS[idx].z * interaction.x) * params.dt - \
                     sqrtGamma * (1 + oldS[idx].z - oldS[idx].y * oldS[idx].y) * rndB + sqrtGamma * oldS[idx].x * oldS[idx].y * rndA;

    bufferQuant.z += ( -params.Gamma * (oldS[idx].z + 1) + 2*params.Jx * oldS[idx].y * interaction.x  - 2*params.Jy * oldS[idx].x * interaction.y) * params.dt - \
                     -sqrtGamma * oldS[idx].x * (1 + oldS[idx].z) * rndA + sqrtGamma * oldS[idx].y * (1 + oldS[idx].z) * rndB;



    rngStates[idx] = localState;
    newS[idx] = bufferQuant;
};
 


void runDissipXY_NN(float *newBuffQuantity, float *oldBuffQuantity,
             PhysicalParams params, KernelParams kernelParams)
{
    simulate<<<kernelParams.numBlocks, kernelParams.blockSize, kernelParams.sharedMemSize>>>(
        (float2 *)newBuffQuantity, (float2 *)oldBuffQuantity,
        params, kernelParams.numTiles, kernelParams.rngStates);    
};


void runDissipXY(float *newBuffQuantity, float *oldBuffQuantity,
    PhysicalParams params, KernelParams kernelParams)
{
    simulate_xy_model<<<kernelParams.numBlocks, kernelParams.blockSize, kernelParams.sharedMemSize>>>(
        (float2 *)newBuffQuantity, (float2 *)oldBuffQuantity,
        params, kernelParams.numTiles, kernelParams.rngStates);    
};


void runDissipXYZ(float *newBuffQuantity, float *oldBuffQuantity,
    PhysicalParams params, KernelParams kernelParams)
{
    simulate_xyz_model<<<kernelParams.numBlocks, kernelParams.blockSize, kernelParams.sharedMemSize>>>(
        (float3 *)newBuffQuantity, (float3 *)oldBuffQuantity,
        params, kernelParams.numTiles, kernelParams.rngStates);    
};
