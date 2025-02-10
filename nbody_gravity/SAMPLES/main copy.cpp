#include <iostream>
#include <fstream>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "./include/kernel.cuh"

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

    pos[p++] = 1;//point.x * scale;  // pos.x
    pos[p++] = 1;//point.y * scale;  // pos.y
    pos[p++] = 1;//point.z * scale;  // pos.z
    pos[p++] = 1.0f;             // mass

    vel[v++] = 1;//velocity.x * vscale;  // pos.x
    vel[v++] = 1;//velocity.y * vscale;  // pos.x
    vel[v++] = 1;//velocity.z * vscale;  // pos.x

    if (vec4vel) vel[v++] = 1.0f;  // inverse mass

    i++;
    };
}; 

int main(){
    int repeats = 50;
    int nBodies = 128;
    int blockSize = 256;
    int numBlocks = 4;
    // float E2 = 0.1f;
    
    unsigned int memRead = 0;
    unsigned int memWrite = 1;

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

    allocateMemory(mem_devicePosition, nBodies);
    allocateMemory(mem_deviceVelocity, nBodies);   
    // cudaMalloc((void**)&mem_devicePosition[0], 4 * sizeof(float)*nBodies);
    // cudaMalloc((void**)&mem_devicePosition[1], 4 * sizeof(float)*nBodies);

    // cudaMalloc((void**)&mem_deviceVelocity[0], 4 * sizeof(float)*nBodies);
    // cudaMalloc((void**)&mem_deviceVelocity[1], 4 * sizeof(float)*nBodies);

    // cudaMemcpy(mem_devicePosition[memWrite], velocities, nBodies*4*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(mem_deviceVelocity[memWrite], positions, nBodies*4*sizeof(float), cudaMemcpyHostToDevice);

    copyMemoryToDevice(mem_devicePosition[memWrite], velocities, nBodies);
    copyMemoryToDevice(mem_deviceVelocity[memWrite], positions, nBodies);


    struct timeval start, stop;
    double time;
    runNbody(
    mem_devicePosition[memWrite], 
    mem_deviceVelocity[memWrite],
    mem_devicePosition[memRead], 
    mem_deviceVelocity[memRead],
    0.1f, 0.1f, nBodies, numBlocks, blockSize);

    gettimeofday(&start, NULL);

    for (int i = 0; i < repeats; i++)
    {
        runNbody(
        mem_devicePosition[memWrite], 
        mem_deviceVelocity[memWrite],
        mem_devicePosition[memRead], 
        mem_deviceVelocity[memRead],
        0.1f, 0.1f, nBodies, numBlocks, blockSize);

        std::swap(memRead, memWrite);
    }

    cudaMemcpy(posGPU, mem_devicePosition[memRead], 4*nBodies*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(velGPU, mem_deviceVelocity[memRead], 4*nBodies*sizeof(float), cudaMemcpyDeviceToHost);
    // copyMemoryToHost(velGPU, mem_deviceVelocity[memRead],nBodies);
    // copyMemoryToHost(posGPU, mem_devicePosition[memRead],nBodies);
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
        printf("%d GPU_vel=%f INPUT_vel=%f\n",i, velGPU[i], velocities[i]);
    }

    printf("N=%d , glops=%f  time=%f\n", nBodies, gflops, time);


    // deleteMemory(mem_devicePosition);
    // deleteMemory(mem_deviceVelocity);

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


// 127 GPU_pos=10.000000 INPUT_pos=1.000000
// 127 GPU_vel=0.000000 INPUT_vel=10.000000


// 127 GPU_pos=0.000000 INPUT_pos=1.000000
// 127 GPU_vel=1.000000 INPUT_vel=10.000000


// 127 GPU_pos=10.000000 INPUT_pos=1.000000
// 127 GPU_vel=0.000000 INPUT_vel=10.000000
