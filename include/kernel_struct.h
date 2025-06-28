#ifndef __KERNEL_STRUCT__
#define __KERNEL_STRUCT__

#define BLOCK_SIZE_DEF 256;
#define USE_HOST_MEM 1

#include <curand_kernel.h>

struct PhysicalParams{
    
	float Jx;
    float Jy;
    float Jz;  
    float dt;
	float Gamma;
    long steps;
	int saveStartPoint;
	int numBodies;
    int subSystemSize;
	int saveStep;
    int seed = time(NULL) + 1;
    int model;
};


struct GPUDev{

    int blockSize = BLOCK_SIZE_DEF;
    int useHostMem = USE_HOST_MEM;
};


struct KernelParams{

    curandState *rngStates;
    int numBlocks;
    int sharedMemSize;
    int numTiles;
    int blockSize;
};

#endif 
