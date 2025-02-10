#ifndef __NBODY_SIMULATION__
#define __NBODY_SIMULATION__
// #pragma once 

class NBodySimulation{

    public:
        NBodySimulation(int numBodies) {nBodies = numBodies; isInitialized=false;};
        virtual ~NBodySimulation() {};
        // DeviceTimer devTimer;

        enum PhysicalQuantity{
            POSITION,
            VELOCITY,
        };

        virtual void run(float) = 0;
        virtual void setPhysicalParams(float, float) = 0;
        
        virtual float* readMemory(PhysicalQuantity) = 0;
        virtual void writeMemory(PhysicalQuantity, float*) = 0;

    protected:
        NBodySimulation() {};

        int nBodies;
        bool isInitialized;
        virtual void _initialize(int) = 0;
        virtual void _summarize() = 0;
};

#endif 