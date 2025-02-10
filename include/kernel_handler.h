#ifndef __KERNEL_HANDLER__
#define __KERNEL_HANDLER__
#include "nbody_simulation.h"
// #include "kernel.cuh"

void runNbody(float *newBuffQuantity, float *oldBuffQuantity,
                        PhysicalParams input, int step, int nBodies, int blockSize)
void copyMemoryToDevice(float *host, float *device, int numBodies);
void copyMemoryToHost(float *host, float *device, int numBodies);
void allocateMemory(float *data[2], int numBodies);

class KernelHandler : public NBodySimulation{

    public:
        KernelHandler(PhysicalParams, unsigned int, unsigned int);
        virtual ~KernelHandler();

        float* mem_hostBuffer;

        virtual void run(int);
        virtual void setPhysicalParams(PhysicalParams);
        virtual float* readMemory();
        virtual void writeMemory(float*);

    protected:
        KernelHandler() {};
        float* mem_deviceBuffer[2];
        PhysicalParams systemParams;

        unsigned int memRead;
        unsigned int memWrite;

        unsigned int d_blockSize;
        unsigned int d_numBlocks;

        virtual void _initialize();
        virtual void _summarize();

 //        float m_damping;
 //        float e2;


};


#include "kernel_handler_impl.h"

#endif
