#include "nbodyCPU.h"
#include "nbody_simulation.h"
#include <math.h>
#include <memory.h>
#include <algorithm>
#include <omp.h>


/*double CPUmethods::randn()
{
    
    double u1 = rand() / (double)RAND_MAX; // Uniform random number between 0 and 1
    double u2 = rand() / (double)RAND_MAX; // Uniform random number between 0 and 1
    while(u1==0){
      u1 = rand() / (double)RAND_MAX; // Uniform random number between 0 and 1 and avoid 0
    }
    // Box-Muller transform
    double z0 = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2); // Normally distributed random number
    // double z1 = sqrt(-2 * log(u1)) * sin(2 * M_PI * u2); // Second normally distributed random number (optional)

    return z0;
}
    
void CPUmethods::randomizeSystem(float *inputBuffer, int numBodies)
{
    int buffIdx = 0, i = 0;
    while (i < numBodies)
    {
        float2CPU sample;

        sample.x = rand() / (float) RAND_MAX * 2 * 3.141;
        sample.y = rand() / (float) RAND_MAX * 2 * 3.141;


        samplesBuffer[buffIdx++] = sample.x;
        samplesBuffer[buffIdx++] = sample.y;

        i++;
    }
}
*/

float CPUmethods::cot(float x)
{
  return 1/tan(x);
}

float CPUmethods::csc(float x)
{
  return 1/sin(x);
}

float CPUmethods::randn()
{
    
    float u1 = rand() / (float)RAND_MAX; // Uniform random number between 0 and 1
    float u2 = rand() / (float)RAND_MAX; // Uniform random number between 0 and 1
    while(u1==0){
      u1 = rand() / (double)RAND_MAX; // Uniform random number between 0 and 1 and avoid 0
    }
    // Box-Muller transform
    float z0 = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2); // Normally distributed random number
    // double z1 = sqrt(-2 * log(u1)) * sin(2 * M_PI * u2); // Second normally distributed random number (optional)

    return z0;
}

NbodySimulationCPU::NbodySimulationCPU(PhysicalParams params) : NBodySimulation(params)
{
    memRead = 0;
    memWrite = 1;
    systemParams = params;
    
    _initialize();
}

NbodySimulationCPU::~NbodySimulationCPU()
{
    _summarize();
    nBodies = 0;
}

void NbodySimulationCPU::run(int step)
{
    simulate();
    std::swap(memRead, memWrite);
}

float *NbodySimulationCPU::readMemory()
{
    return mem_Buffer;
}

void NbodySimulationCPU::writeMemory(float *data)
{
    // float *inData = 0;
    memcpy(mem_Buffer, data, 2*nBodies*sizeof(float));
}

void NbodySimulationCPU::_initialize()
{
    nBodies = systemParams.numBodies;
    mem_Buffer = new float[2*nBodies];
    memset(mem_Buffer, 0, 2*nBodies*sizeof(float));
}

void NbodySimulationCPU::_summarize()
{
    delete [] mem_Buffer;
}

void NbodySimulationCPU::simulate()
{
//    #ifdef OPENMP
//    #pragma omp parallel for
//    #endif

    float sqr3 = sqrt(3);
    float oldBufferQuantity[2*nBodies];
    memcpy(oldBufferQuantity, mem_Buffer, 2*nBodies * sizeof(float));


    for (int i = 0; i < nBodies; ++i)
    {
        int idx = 2*i;
        float buffer[2];

        buffer[0] = mem_Buffer[idx+0];
        buffer[1] = mem_Buffer[idx+1];
        

        float3CPU interaction = computeInteraction(oldBufferQuantity);

        buffer[0] += (- 2 * systemParams.Jx * sin(oldBufferQuantity[idx+1]) * sqr3 * interaction.x\
         + 2 * systemParams.Jy * cos(oldBufferQuantity[idx+1]) * sqr3 * interaction.y ) * systemParams.dt +\
         2 * systemParams.Gamma * (cpuMet.cot(oldBufferQuantity[idx+0]) - cpuMet.csc(oldBufferQuantity[idx+0])/sqr3) * systemParams.dt;

        buffer[1] += (- 2 * systemParams.Jx * cpuMet.cot(oldBufferQuantity[idx+0]) * cos(oldBufferQuantity[idx+1]) * sqr3 * interaction.x \
        - 2 * systemParams.Jy * cpuMet.cot(oldBufferQuantity[idx+0]) * sin(oldBufferQuantity[idx+1]) * sqr3 * interaction.y +\
         2 * systemParams.Jz * sqr3 * interaction.z) * systemParams.dt +  sqrt(2) \
         * sqrt(systemParams.Gamma) * sqrt(1 + 2*cpuMet.cot(oldBufferQuantity[idx+0])*cpuMet.cot(oldBufferQuantity[idx+0]) -\
          2*cpuMet.cot(oldBufferQuantity[idx+0])*cpuMet.csc(oldBufferQuantity[idx+0])/sqr3) * sqrt(systemParams.dt) * cpuMet.randn(); 

        mem_Buffer[idx+0] = buffer[0];
        mem_Buffer[idx+1] = buffer[1];
    }
}

float3CPU NbodySimulationCPU::computeInteraction(float* oldBuffQuantity)
//#ifdef OPENMP
//#pragma omp parallel for
//#endif
{
    float3CPU interaction = {0.0, 0.0, 0.0};
    float J;
        
    // We unroll this loop 4X for a small performance boost.
    int idx = 0;
    while (idx < nBodies) 
    {
        interaction.x += sin(oldBuffQuantity[2*idx+0]) * cos(oldBuffQuantity[2*idx+1]);
        interaction.y += sin(oldBuffQuantity[2*idx+0]) * sin(oldBuffQuantity[2*idx+1]);
        interaction.z += cos(oldBuffQuantity[2*idx+0]);
        idx++;
    }
    return interaction;
}
