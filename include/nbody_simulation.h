#ifndef __NBODY_SIMULATION__
#define __NBODY_SIMULATION__

#define BLOCK_SIZE_DEF 256;
#define USE_HOST_MEM 1

#include <curand_kernel.h>

struct PhysicalParams
{
	float Jx;
    float Jy;
    float Jz;  
    float dt;
	float Gamma;
    long steps;
	int saveStartPoint;
	int numBodies;    
	int saveStep;
    int seed = time(NULL) + 1;
};

struct GPUDev{

    int blockSize = BLOCK_SIZE_DEF;
    int useHostMem = USE_HOST_MEM;
};

struct KernelParams{

    int numBlocks;
    int sharedMemSize;
    int numTiles;
    int blockSize;
    curandState *rngStates;
};

class NBodySimulation{

    public:
        NBodySimulation(PhysicalParams params) {nBodies = params.numBodies; isInitialized=false;};
        virtual ~NBodySimulation() {};

        virtual void run() = 0;
        virtual float* readMemory() = 0;
        virtual void writeMemory(float*) = 0;

    protected:
        NBodySimulation() {};

        int nBodies;
        bool isInitialized;

        virtual void _initialize() = 0;
        virtual void _summarize() = 0;
};

#endif 
