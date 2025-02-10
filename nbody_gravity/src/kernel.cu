#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "../include/nbody_simulation.h"
namespace cg = cooperative_groups;

#define BLOCKDIM 256
// #define modWrap(x,m) (((x)<m)?(x):(x-m))
// #define SX(i) globalPosition[i+blockDim.x*threadIdx.y]
// #define SX_SUM(i,j) globalPosition[i+blockDim.x*j]

__constant__ float E2 = 1.0f;

__device__ float3 interact(float3 acci, float4 ri, float4 rj)
{   
    float3 r;
    
    r.x = ri.x - rj.x;
    r.y = ri.y - rj.y;
    r.z = ri.z - rj.z;

    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + E2;
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = 1.0f/sqrtf(distSixth);
    float s = rj.w * invDistCube;

    acci.x += r.x * s;
    acci.y += r.y * s;
    acci.z += r.z * s;

    return acci;   
}

//__device__ float3 computeForce(float4 bodyPositions, float4* positions, int numBodies)

__device__ float3 computeForce(float4 bodyPositions, float4* positions, int numTiles, cg::thread_block cta) 
{
  extern __shared__ float4 globalPosition[];
  float3 acceleration = {0.0f, 0.0f, 0.0f};
  for (int tile = 0; tile < numTiles; tile++) 
  {
        globalPosition[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];
        cg::sync(cta);

        #pragma unroll 128
        for (unsigned int counter = 0; counter < blockDim.x; counter++) 
        {
            acceleration = interact(acceleration, bodyPositions, globalPosition[counter]);
        };
        cg::sync(cta);
  }

  return acceleration;
}

// template <bool BODY_MULTITHREAD>
 __global__ void simulate(float4 *newPosition, float4 *oldPosition,
                        float4 *velocity,
                        float dt, float  damping, int numTiles)
{

    cg::thread_block cta = cg::this_thread_block();

    int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    float4 position = oldPosition[idx];
    float3 acceleration = computeForce(position, oldPosition, numTiles, cta);
    float4 intVelocity = velocity[idx];

    intVelocity.x += acceleration.x * dt;
    intVelocity.y += acceleration.y * dt;
    intVelocity.z += acceleration.z * dt;  

    intVelocity.x *= damping;
    intVelocity.y *= damping;
    intVelocity.z *= damping;
        
    position.x += intVelocity.x * dt;
    position.y += intVelocity.y * dt;
    position.z += intVelocity.z * dt;
    
    newPosition[idx] = position;
    velocity[idx] = intVelocity;
}

void runNbody(float *newPosition, float *oldPosition, float *velocity,
                        float dt, float damping, int nBodies, int blockSize){
    
    int numBlocks = (nBodies + blockSize - 1) / blockSize;
    int sharedMemSize = blockSize * 4 * sizeof(float);
    int numTiles = (nBodies + blockSize - 1) / blockSize;

    simulate<<<numBlocks, blockSize, sharedMemSize>>>(
        (float4 *)newPosition, (float4 *)oldPosition,
        (float4 *)velocity,
        dt, damping, numTiles);    
};

void copyMemoryToHost(float *host, float *device, int numBodies){
    cudaMemcpy(host, device, numBodies*4*sizeof(float), cudaMemcpyDeviceToHost);
};

void copyMemoryToDevice(float *host, float *device, int numBodies){
    cudaMemcpy(host, device, numBodies*4*sizeof(float), cudaMemcpyHostToDevice);
};

void allocateMemory(float *data[2], int numBodies){
    unsigned int memorySize = sizeof(float) * 4 * numBodies;
    cudaMalloc((void**)&data[0], memorySize);
    cudaMalloc((void**)&data[1], memorySize);
}

void deleteMemory(float *velocity[2]){
    if(velocity[0])
        cudaFree((void**)&velocity[0]);
    if(velocity[1])
        cudaFree((void**)&velocity[1]);
};   
float setE2(float e2){
    return e2*e2;
};

void runNbody(float *newPosition, float *oldPosition, float *velocity,
                        float dt, float damping, int nBodies, int blockSize);
void copyMemoryToDevice(float *host, float *device, int numBodies);
void copyMemoryToHost(float *host, float *device, int numBodies);
void allocateMemory(float *velocity[2], int numBodies);
void deleteMemory(float *velocity[2]);
float setE2(float e2);

// __device__ float3 getTile(float4 position, float3 acceleration)
// {
//     extern __shared__ float4 globalPosition[];
//     #pragma unroll 128
//     for(int i=0; i < blockDim.x; i++){
//         acceleration = interact(acceleration, SX(i), position); i+=1;
//     };
//     return acceleration;
// }

// // template <bool BODY_MULTITHREAD>
//  __device__ float3 computeForce(float4 bodyPositions, float4* positions, int numBodies)
//  {
//     extern __shared__ float4 globalPosition[];
//     float3 acceleration = {0.0f, 0.0f, 0.0f};

//     int p = blockDim.x;
//     int q = blockDim.y;
//     int n = numBodies;

//     int start = n/q * threadIdx.y;
//     int tile_init = start/(n/q);
//     int tile = tile_init;
//     int stop = start + n/q;

//     for (int i = start; i < stop; i += p, tile++) 
//     {
//         globalPosition[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];
//         // __syncthreads();
//         acceleration = getTile(bodyPositions, acceleration);
//         // __syncthreads();

//         cg::sync(cta);

//     }
//     return acceleration;
//  };

// RUN BODY MODIF
// if (grid.y == 1)
// simulate<false><<<numBlocks, blockSize, sharedMemSize>>>((float4 *)newPosition, (float4 *)newVelocity,
//                                 (float4 *)oldPosition, (float4 *)oldVelocity,
//                                 dt, damping, nBodies);
// else
//     simulate<true><<<numBlocks,nInteractions,sharedMemSize>>>((float4 *)newPosition, (float4 *)newVelocity,
//                                     (float4 *)oldPosition, (float4 *)oldVelocity,
//                                     dt, damping, nBodies);

// SIMULATE MODIF
// float3 acceleration = computeForce<BODY_MULTITHREAD>(position, oldPosition, nBodies);

// COMP_FORCE MODIF
// globalPosition[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];
// globalPosition[threadIdx.x+blockDim.x*threadIdx.y] = BODY_MULTITHREAD ?
// positions[(modWrap(blockDim.x + tile, gridDim.x)*blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x]:
// positions[modWrap(blockIdx.x+tile, gridDim.x) * blockDim.x + threadIdx.x];

// __syncthreads();
// acceleration = getTile(bodyPositions, acceleration);
// __syncthreads();

// if (BODY_MULTITHREAD){

//     SX_SUM(threadIdx.x, threadIdx.y).x = acceleration.x;
//     SX_SUM(threadIdx.x, threadIdx.y).y = acceleration.y;
//     SX_SUM(threadIdx.x, threadIdx.y).z = acceleration.z;
//     __syncthreads();

//     if(threadIdx.y==0){
//         for (int i = 0; i < blockDim.y; i++)
//         {
//             acceleration.x += SX_SUM(threadIdx.x,i).x;
//             acceleration.y += SX_SUM(threadIdx.x,i).y;
//             acceleration.z += SX_SUM(threadIdx.x,i).z;
//         }
//     }

// }

