#ifndef __KERNEL_HANDLER__
#define __KERNEL_HANDLER__
// #pragma once
#include "nbody_simulation.h"
// #include "kernel.cuh"

void runNbody(float *newPosition, float *oldPosition, float *velocity,
                        float dt, float damping, int nBodies, int blockSize);
void copyMemoryToDevice(float *host, float *device, int numBodies);
void copyMemoryToHost(float *host, float *device, int numBodies);
void allocateMemory(float *velocity[2], int numBodies);
void deleteMemory(float *velocity[2]);
float setE2(float e2);


class   KernelHandler : public NBodySimulation{

    public:
        KernelHandler(int, unsigned int, unsigned int);
        virtual ~KernelHandler();

        float* mem_hostPosition;
        float* mem_hostVelocity;
        
        virtual void run(float);
        virtual void setPhysicalParams(float, float);
        virtual float* readMemory(PhysicalQuantity);
        virtual void writeMemory(PhysicalQuantity, float*);

    protected:
        KernelHandler() {};
        float* mem_devicePosition[2];
        float* mem_deviceVelocity;

        float m_damping;
        float e2;
        
        unsigned int memRead;
        unsigned int memWrite;

        unsigned int d_blockSize;
        unsigned int d_numBlocks;

        virtual void _initialize(int numBodies);
        virtual void _summarize();


};


#include "kernel_handler_impl.h"

#endif