#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <math_constants.h>
#include "../include/nbody_simulation.h"
#include <math.h>
// #include <curand_kernel.h>

#include <cstdio>
namespace cg = cooperative_groups;

void runNbody(float *newBuffQuantity, float *oldBuffQuantity,
    PhysicalParams params, KernelParams kernelParams);
    void initializeRandomStates(KernelParams kernelParams);
void copyMemoryToDevice(float *host, float *device, int numBodies);
void copyMemoryToHost(float *host, float *device, int numBodies);
void allocateMappedMemory(float * data[2], unsigned int memorySize);
void allocateMemory(float *data[2], unsigned int memorySize);
void matchMemory(float *dataDevice[2], float *dataHost[2]);

__device__ float cotf(float x)
{  
    return 1 / tanf(x);
}

__device__ float cscf(float x)
{
    return  1 / sinf(x);
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

// __global__ void generate_kernel(curandState *my_curandstate, const unsigned int n, const unsigned *max_rand_int, const unsigned *min_rand_int,  unsigned int *result){

//     int idx = threadIdx.x + blockDim.x*blockIdx.x;
  
//     int count = 0;
//     while (count < n){
//       float myrandf = curand_uniform(my_curandstate+idx);
//       myrandf *= (max_rand_int[idx] - min_rand_int[idx]+0.999999);
//       myrandf += min_rand_int[idx];
//       int myrand = (int)truncf(myrandf);
  
//       assert(myrand <= max_rand_int[idx]);
//       assert(myrand >= min_rand_int[idx]);
//       result[myrand-min_rand_int[idx]]++;
//       count++;}
//   }
  

static __global__ void setRnStates(curandState *rndStates, unsigned long long seed)
{
    int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    curand_init(seed+idx, idx, 0, &rndStates[idx]);
}

 __global__ void simulate(float2 *newBuffQuantity, float2 *oldBuffQuantity,
                        PhysicalParams params, int numTiles, 
                        curandState *rngStates)
{
    cg::thread_block cta = cg::this_thread_block();
    
    int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    float2 bufferQuant = oldBuffQuantity[idx];
    
    float3 interaction = computeInteraction(oldBuffQuantity, idx, numTiles, cta);
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
}


void runNbody(float *newBuffQuantity, float *oldBuffQuantity,
             PhysicalParams params, KernelParams kernelParams)
{
    simulate<<<kernelParams.numBlocks, kernelParams.blockSize, kernelParams.sharedMemSize>>>(
        (float2 *)newBuffQuantity, (float2 *)oldBuffQuantity,
        params, kernelParams.numTiles, kernelParams.rngStates);    
};


void initializeRandomStates(KernelParams kernelParams)
{
    cudaMemset(
        kernelParams.rngStates, 0, 
        kernelParams.numBlocks * kernelParams.blockSize * sizeof(curandState));

    setRnStates<<<kernelParams.numBlocks, kernelParams.blockSize>>>(
        kernelParams.rngStates, time(NULL));
};


void copyMemoryToHost(float *host, float *device, int numBodies)
{
    cudaMemcpy(host, device, numBodies*2*sizeof(float), cudaMemcpyDeviceToHost);
};


void copyMemoryToDevice(float *host, float *device, int numBodies)
{
    cudaMemcpy(host, device, numBodies*2*sizeof(float), cudaMemcpyHostToDevice);
};


void matchMemory(float *dataDevice[2], float *dataHost[2])
{
   cudaHostGetDevicePointer((void **)&dataDevice[0], (void *)dataHost[0], 0);
   cudaHostGetDevicePointer((void **)&dataDevice[1], (void *)dataHost[1], 0);
};


void allocateMappedMemory(float * data[2], unsigned int memorySize)
{
    cudaHostAlloc((void **)&data[0], memorySize,
        cudaHostAllocMapped | cudaHostAllocPortable); //cudaHostAllocPortable
    cudaHostAlloc((void **)&data[1], memorySize,
        cudaHostAllocMapped | cudaHostAllocPortable);
}


void allocateMemory(float *data[2], unsigned int memorySize)
{
    cudaMalloc((void**)&data[0], memorySize);
    cudaMalloc((void**)&data[1], memorySize);
}
