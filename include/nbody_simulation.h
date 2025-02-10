#ifndef __NBODY_SIMULATION__
#define __NBODY_SIMULATION__
// #pragma once 

#define MAX_FILENAME_LENGTH 128

struct PhysicalParams
{
    int numBodies;
    int systemType;
    double dt;
    long steps;
    int savesteps;
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

class NBodySimulation{

    public:
        NBodySimulation(int numBodies) {nBodies = numBodies; isInitialized=false;};
        virtual ~NBodySimulation() {};

        virtual void run(int) = 0;
        virtual float* readMemory() = 0;
        virtual void writeMemory(float*) = 0;
         //        virtual void setPhysicalParams(float, float) = 0;

    protected:
        NBodySimulation() {};

        int nBodies;
        bool isInitialized;
        virtual void _initialize() = 0;
        virtual void _summarize() = 0;
};

#endif 
