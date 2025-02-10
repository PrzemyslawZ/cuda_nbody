#ifndef __NBODY_CPU__
#define __NBODY_CPU__
// #pragma once 

#include "nbody_simulation.h"

struct float3CPU{
    float x,y,z;
};

class CPUmethods{
    friend class NbodySimulationCPU;

    public:
        CPUmethods() {};
        virtual ~CPUmethods() {};
        int A = 0;
        void randomizeSystem(float*,float*,float,float,int);

    private:
        float3CPU scaleVector(float3CPU &, float);
        float normalize(float3CPU &);
        float vecDot(float3CPU vec1, float3CPU vec2);
        float3CPU crossDot(float3CPU vec1, float3CPU vec2);

};

class NbodySimulationCPU : public NBodySimulation{

    public:
        NbodySimulationCPU(int);
        virtual ~NbodySimulationCPU();

        virtual void run(float);
        virtual void setPhysicalParams(float, float);
        virtual float* readMemory(PhysicalQuantity);
        virtual void writeMemory(PhysicalQuantity, float*);

    protected:

        NbodySimulationCPU() {};
        float* mem_Position;
        float* mem_Velocity;
        float* mem_Force;

        float m_damping;
        float m_e2;
        
        unsigned int memRead;
        unsigned int memWrite;

        // unsigned int timer;

        virtual void _initialize(int numBodies);
        virtual void _summarize();

        void simulate(float);
        void computeForce();
        void interact(float acceleration[3], float position1[4], float position2[4], float e2);


};

#include "nbodyCPU_impl.h"
#endif
