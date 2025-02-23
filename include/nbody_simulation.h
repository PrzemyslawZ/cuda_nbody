#ifndef __NBODY_SIMULATION__
#define __NBODY_SIMULATION__
// #pragma once 

#define MAX_FILENAME_LENGTH 128
#define BLOCK_SIZE_DEF 256;
#define NUM_BLOCKS_DEF 4
#define USE_HOST_MEM 1

#include <curand_kernel.h>

struct PhysicalParams
{
    int numBodies;
    // int systemType;
    float dt;
    long steps;
    int saveStep;
    int Nx_spins;
    int Ny_spins;
    int Nz_spins;
    // float Gamma1;
    // float Gamma2;
    // float Nth1;
    // float Nth2;
    float Gamma;
    // float GammaPhase;
    float Jx;
    float Jy;
    float Jz;
    // float Omegax;
    // float Omegay;
    // float Omegaz;
    // float OmegaD;  
    float saveStartPoint;
    int seed = time(NULL) + 1;
};

struct GPUDev{

    int blockSize = BLOCK_SIZE_DEF;
    int numBlocks = NUM_BLOCKS_DEF;
    int useHostMem = USE_HOST_MEM;
};

struct KernelParams{

    int numBlocks;
    int sharedMemSize;
    int numTiles;
    int blockSize;
    curandState *rngStates;
    // curandState *rngHostStates;
};

class NBodySimulation{

    public:
        NBodySimulation(PhysicalParams params) {nBodies = params.numBodies; isInitialized=false;};
        virtual ~NBodySimulation() {};

        virtual void run(int) = 0;
        virtual float* readMemory() = 0;
        virtual void writeMemory(float*) = 0;

    protected:
        NBodySimulation() {};

        int nBodies;
        bool isInitialized;;
        virtual void _initialize() = 0;
        virtual void _summarize() = 0;
};

#endif 
