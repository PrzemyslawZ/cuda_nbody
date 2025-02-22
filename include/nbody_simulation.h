#ifndef __NBODY_SIMULATION__
#define __NBODY_SIMULATION__
// #pragma once 

#define MAX_FILENAME_LENGTH 128
#define BLOCK_SIZE_DEF 256;
#define NUM_BLOCKS_DEF 4

struct PhysicalParams
{
    int numBodies;
    int systemType;
    double dt;
    long steps;
    int saveStep;
    int Nx_spins;
    int Ny_spins;
    int Nz_spins;
    double Gamma1;
    double Gamma2;
    double Nth1;
    double Nth2;
    double Gamma;
    double GammaPhase;
    double Jx;
    double Jy;
    double Jz;
    double Omegax;
    double Omegay;
    double Omegaz;
    double OmegaD;  
    double startMeasuring;
    char filename[MAX_FILENAME_LENGTH];
    char directory[MAX_FILENAME_LENGTH];
    int ThreadId;
};

struct GPUDev{

    int blockSize = BLOCK_SIZE_DEF;
    int numBlocks = NUM_BLOCKS_DEF;
};

class NBodySimulation{

    public:
        NBodySimulation(PhysicalParams *params) {nBodies = params->numBodies; isInitialized=false;};
        virtual ~NBodySimulation() {};

        virtual void run(int) = 0;
        virtual float* readMemory() = 0;
        virtual void writeMemory(float*) = 0;
        virtual void setPhysicalParams(PhysicalParams *) = 0;

    protected:
        NBodySimulation() {};

        int nBodies;
        bool isInitialized;
        virtual void _initialize() = 0;
        virtual void _summarize() = 0;
};

#endif 
