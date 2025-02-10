#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
// Include associated header file.
// #include "../include/kernel.cuh"
// #define threadsPerBlock 64
#define modWrap(x,m) (((x)<m)?(x):(x-m))
#define SX(i) globalPosition[i+blockDim.x*threadIdx.y]
#define SX_SUM(i,j) globalPosition[i+blockDim.x*j]

#define BLOCKDIM 256
// #define LOOP_UNROLL 128

__constant__ float E2 = 0.1f;

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


__device__ float3 getTile(float4 position, float3 acceleration)
{
    extern __shared__ float4 globalPosition[];
    #pragma unroll 128
    for(int i=0; i < blockDim.x; i++){
        acceleration = interact(acceleration, SX(i), position); i+=1;
    };
    return acceleration;
}

 __device__ float3 computeForce(float4 bodyPositions, float4* positions, int numBodies)
 {
    extern __shared__ float4 globalPosition[];
    float3 acceleration = {0.0f, 0.0f, 0.0f};

    int p = blockDim.x;
    int q = blockDim.y;
    int n = numBodies;

    int start = n/q * threadIdx.y;
    int tile_init = start/(n/q);
    int tile = tile_init;
    int stop = start + n/q;

    for (int i = start; i < stop; i += p, tile++) 
    {
        globalPosition[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];
        // __syncthreads();
        acceleration = getTile(bodyPositions, acceleration);
        // __syncthreads();

    }
    return acceleration;
 };

 __global__ void simulate(float4 *newPosition, float4 *newVelocity,
                        float4 *oldPosition, float4 *oldVelocity,
                        float dt, float  damping, int nBodies)
{
    // extern __shared__ float4 globalPosition[];
    int idx = __mul24(blockIdx.x , blockDim.x) + threadIdx.x;
    float4 position = oldPosition[idx];
    float3 acceleration = computeForce(position, oldPosition, nBodies);

    float4 velocity = oldVelocity[idx];

    velocity.x += acceleration.x * dt;
    velocity.y += acceleration.y * dt;
    velocity.z += acceleration.z * dt;  

    velocity.x *= damping;
    velocity.y *= damping;
    velocity.z *= damping;
        
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;
    position.z += velocity.z * dt;
    
    newPosition[idx] = position;
    newVelocity[idx] = velocity;
    // printf("%d\n", idx);

}

inline float dot(float3 v0, float3 v1) {
  return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
};

void randomizeBodies(float *pos, float *vel,
                     float clusterScale, float velocityScale, int numBodies,
                     bool vec4vel)
{
    float scale = clusterScale * std::max<float>(1.0f, numBodies / (1024.0f));
    float vscale = velocityScale * scale;

    int p = 0, v = 0;
    int i = 0;

    while (i < numBodies) {
    float3 point;
    // const int scale = 16;
    point.x = rand() / (float)RAND_MAX * 2 - 1;
    point.y = rand() / (float)RAND_MAX * 2 - 1;
    point.z = rand() / (float)RAND_MAX * 2 - 1;
    float lenSqr = dot(point, point);

    if (lenSqr > 1) continue;

    float3 velocity;
    velocity.x = rand() / (float)RAND_MAX * 2 - 1;
    velocity.y = rand() / (float)RAND_MAX * 2 - 1;
    velocity.z = rand() / (float)RAND_MAX * 2 - 1;
    lenSqr = dot(velocity, velocity);

    if (lenSqr > 1) continue;

    pos[p++] = point.x * scale;  // pos.x
    pos[p++] = point.y * scale;  // pos.y
    pos[p++] = point.z * scale;  // pos.z
    pos[p++] = 1.0f;             // mass

    vel[v++] = velocity.x * vscale;  // pos.x
    vel[v++] = velocity.y * vscale;  // pos.x
    vel[v++] = velocity.z * vscale;  // pos.x

    if (vec4vel) vel[v++] = 1.0f;  // inverse mass

    i++;
    };
};



void runNbody(float *newPosition, float *newVelocity,
                        float *oldPosition, float *oldVelocity,
                        float dt, float  damping, int nBodies, 
                        int numBlocks, int blockSize){
    
    int sharedMemSize = blockSize * 4 * sizeof(float);  // 4 floats for pos
    simulate<<<numBlocks, blockSize, sharedMemSize>>>((float4 *)newPosition, (float4 *)newVelocity,
                                    (float4 *)oldPosition, (float4 *)oldVelocity,
                                    dt, damping, nBodies);
};

int main(){

    int repeats = 300;
    int nBodies = 128;
    int blockSize = 256;
    int numBlocks = 4;

    unsigned int memRead = 0;
    unsigned int memWrite =1;

    float* mem_devicePosition[2];
    float* mem_deviceVelocity[2];

    float* positions = new float[nBodies*4];
    float* velocities = new float[nBodies*4];
    
    float *posGPU = new float[nBodies*4];
    float *velGPU = new float[nBodies*4];

    mem_devicePosition[0] = mem_devicePosition[1] = 0;
    mem_deviceVelocity[0] = mem_deviceVelocity[1] = 0;

    randomizeBodies(positions, velocities,
                1.52f, 2.f,
                nBodies, true);

    cudaMalloc((void**)&mem_devicePosition[0], 4 * sizeof(float)*nBodies);
    cudaMalloc((void**)&mem_devicePosition[1], 4 * sizeof(float)*nBodies);

    cudaMalloc((void**)&mem_deviceVelocity[0], 4 * sizeof(float)*nBodies);
    cudaMalloc((void**)&mem_deviceVelocity[1], 4 * sizeof(float)*nBodies);

    cudaMemcpy(mem_devicePosition[memWrite], positions, nBodies*4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mem_deviceVelocity[memWrite], velocities, nBodies*4*sizeof(float), cudaMemcpyHostToDevice);

    struct timeval start, stop;
    double time;

    runNbody(
    mem_devicePosition[memWrite], 
    mem_deviceVelocity[memWrite],
    mem_devicePosition[memRead], 
    mem_deviceVelocity[memRead],
    0.001f, 0.1f, nBodies, numBlocks, blockSize);

    // std::swap(memRead, memWrite);

    gettimeofday(&start, NULL);

    for (int i = 0; i < repeats; i++)
    {
        runNbody(
        mem_devicePosition[memWrite], 
        mem_deviceVelocity[memWrite],
        mem_devicePosition[memRead], 
        mem_deviceVelocity[memRead],
        0.001f, 0.1f, nBodies, numBlocks, blockSize);

        std::swap(memRead, memWrite);
    }
    cudaMemcpy(posGPU, mem_devicePosition[memRead], 4*nBodies*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(velGPU, mem_deviceVelocity[memRead], 4*nBodies*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gettimeofday(&stop, NULL);

    long seconds  = stop.tv_sec  - start.tv_sec;
    long useconds = stop.tv_usec - start.tv_usec;

    time = seconds * 1e3 + useconds * 1e-3;

    const int flopsPerInteraction = 20;
    double interactionsPerSecond = (float)nBodies * (float)nBodies;
    interactionsPerSecond *= 1e-9 * repeats * 1000 / time;
    double gflops = interactionsPerSecond * (float)flopsPerInteraction;


    for (int i = 0; i < nBodies; i++)
    {
        printf("%d GPU_pos=%f INPUT_pos=%f\n",i, posGPU[i], positions[i]);
        // printf("%d GPU_vel=%f INPUT_vel=%f\n",i, velGPU[i], velocities[i]);
    }

    printf("N=%d , glops=%f  time=%f\n", nBodies, gflops, time);
    cudaFree(mem_devicePosition[memWrite]);
    cudaFree(mem_deviceVelocity[memWrite]);

    cudaFree(mem_devicePosition[memRead]);
    cudaFree(mem_deviceVelocity[memRead]);

    free(positions);
    free(velocities);
    free(posGPU);
    free(velGPU);
    return 0;
}


