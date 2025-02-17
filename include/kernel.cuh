#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define modWrap(x,m) (((x)<m)?(x):(x-m))
#define SX(i) globalPosition[i+blockDim.x*threadIdx.y]
#define SX_SUM(i,j) globalPosition[i+blockDim.x*j]

void runNbody(float *newBuffQuantity, float *oldBuffQuantity,
    PhysicalParams *params, int step, int nBodies, int blockSize);

//void copyMemoryToDevice(float *host, float *device, int numBodies);
//void copyMemoryToHost(float *host, float *device, int numBodies);
void allocateMemory(float *data[2], int numBodies);
void matchMemory(float *dataDevice[2], float *dataHost[2]);
