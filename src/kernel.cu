#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <math_constants.h>
#include "../include/nbody_simulation.h"
#include <math.h>
#include <curand_kernel.h>
namespace cg = cooperative_groups;


#define BLOCKDIM 16

__constant__ float E2 = 1.0f;

__device__ float cotf(float x)
{   float t = tanf(x);
    return (t == 0.0f) ? CUDART_NAN_F : 1.0f / t;
}

__device__ float cscf(float x)
{   float s = sinf(x);
    return (s == 0.0f) ? CUDART_NAN_F : 1.0f / s;
}

__device__ float3 computeInteraction(float2* buffer, int idx, int numTiles, cg::thread_block cta) 
{
  extern __shared__ float2 globalBuffer[];
  float3 interaction = {0.0f, 0.0f, 0.0f};

  for (int tile = 0; tile < numTiles; tile++) 
  {
        globalBuffer[threadIdx.x] = buffer[tile * blockDim.x + threadIdx.x];
        cg::sync(cta);

        #pragma unroll 128
        for (unsigned int counter = 0; counter < blockDim.x; counter++) 
        {
            interaction.x += sinf(globalBuffer[counter].x) * cosf(globalBuffer[counter].y);
            interaction.y += sinf(globalBuffer[counter].x) * sinf(globalBuffer[counter].y);
            interaction.z += cosf(globalBuffer[counter].x);
        };
        cg::sync(cta);
  }
  return interaction;
}

 __global__ void simulate(float2 *newBuffQuantity, float2 *oldBuffQuantity,
                        PhysicalParams params, int step, int numTiles, long long seed)
{
    cg::thread_block cta = cg::this_thread_block();

    int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    float2 buffer = oldBuffQuantity[idx];
    
    float3 interaction = computeInteraction(oldBuffQuantity, idx, numTiles, cta);
    float sqr3 = sqrtf(3);

    curandState state;
    seed += idx;
    curand_init(seed, idx, 0, &state);


    interaction.x += (- 2 * params.Jx * sinf(oldBuffQuantity[idx].y) * sqr3 * interaction.x + 2\
    * params.Jy * cosf(oldBuffQuantity[idx].y) * sqr3 * interaction.y ) * params.dt +  \
    2 * params.Gamma * (cotf(oldBuffQuantity[idx].x) - cscf(oldBuffQuantity[idx].x)/sqr3) \
    * params.dt;

    interaction.y += (- 2 * params.Jx * cotf(oldBuffQuantity[idx].x) * \
    cosf(oldBuffQuantity[idx].y) * sqr3 * interaction.x - \
    2 * params.Jy * cotf(oldBuffQuantity[idx].x) * sinf(oldBuffQuantity[idx].y) \
    * sqr3 * interaction.y + 2 * params.Jz * sqr3 * interaction.z) * params.dt; 

    newBuffQuantity[idx] = interaction;
}

void runNbody(float *newBuffQuantity, float *oldBuffQuantity,
                        PhysicalParams params, int step, int nBodies, int blockSize)
{
    int numBlocks = (nBodies + blockSize - 1) / blockSize;
    int sharedMemSize = blockSize * 2 * sizeof(float);
    int numTiles = (nBodies + blockSize - 1) / blockSize;
    unsigned long long seed = static_cast<unsigned long long>(time(NULL)) * (step+1);

    simulate<<<numBlocks, blockSize, sharedMemSize>>>(
        (float2 *)newBuffQuantity, (float2 *)oldBuffQuantity,
        params, step, numTiles, seed);    
};


void copyMemoryToHost(float *host, float *device, int numBodies)
{
    cudaMemcpy(host, device, numBodies*2*sizeof(float), cudaMemcpyDeviceToHost);
};


void copyMemoryToDevice(float *host, float *device, int numBodies)
{
    cudaMemcpy(host, device, numBodies*2*sizeof(float), cudaMemcpyHostToDevice);
};


void allocateMemory(float *data[2], int numBodies)
{
    unsigned int memorySize = sizeof(float) * 2 * numBodies;
    cudaMalloc((void**)&data[0], memorySize);
    cudaMalloc((void**)&data[1], memorySize);
}


void runNbody(float *newBuffQuantity, float *oldBuffQuantity,
                        PhysicalParams params, int step, int nBodies, int blockSize)

void copyMemoryToDevice(float *host, float *device, int numBodies);
void copyMemoryToHost(float *host, float *device, int numBodies);
void allocateMemory(float *data[2], int numBodies);
