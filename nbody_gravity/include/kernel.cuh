#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define modWrap(x,m) (((x)<m)?(x):(x-m))
#define SX(i) globalPosition[i+blockDim.x*threadIdx.y]
#define SX_SUM(i,j) globalPosition[i+blockDim.x*j]


void runNbody(float *newBuffQuantity, float *oldBuffQuantity, float *velocity,
                        float dt, float damping, int nBodies, int blockSize);

void copyMemoryToDevice(float *host, float *device, int numBodies);
void copyMemoryToHost(float *host, float *device, int numBodies);
void allocateMemory(float *velocity[2], int numBodies);
void deleteMemory(float *velocity[2]);
