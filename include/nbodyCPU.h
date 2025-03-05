#ifndef __NBODY_CPU__
#define __NBODY_CPU__
// #pragma once 

#include "nbody_simulation.h"

struct float2CPU{
    float x,y;
};

struct float3CPU{
    float x,y,z;
};

class CPUmethods{
    friend class NbodySimulationCPU;

    public:
        CPUmethods() {};
        virtual ~CPUmethods() {};

    private:
        float randn();
        float cot(float);
        float csc(float);
};

class NbodySimulationCPU : public NBodySimulation{

    public:
        NbodySimulationCPU(PhysicalParams);
        virtual ~NbodySimulationCPU();

        virtual void run();
        virtual float* readMemory();
        virtual void writeMemory(float*);

    protected:
        NbodySimulationCPU() {};
        CPUmethods cpuMet;
        
        float* mem_Buffer;
        PhysicalParams systemParams;
        
        unsigned int memRead;
        unsigned int memWrite;

        virtual void _initialize();
        virtual void _summarize();

        void simulate();
        float3CPU computeInteraction(float*);
        float interact(int id1, int id2);
};

#include "nbodyCPU_impl.h"
#endif
