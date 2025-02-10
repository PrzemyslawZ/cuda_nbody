#include "nbodyCPU.h"
#include "nbody_simulation.h"
#include <math.h>
#include <memory.h>
#include <algorithm>
#include <omp.h>

void CPUmethods::randomizeSystem(float *positions, float *velocities,
                                 float sizeScale, float velocityScale, 
                                 int numBodies)
{
    float scale = sizeScale * std::max(1.0f, numBodies / (1024.f));
    float velScale = velocityScale * scale;

    int posIdx = 0, velPtr = 0, i = 0;
    while (i < numBodies)
    {
        float3CPU point;
        float3CPU velocity;

        point.x = rand() / (float) RAND_MAX * 2 - 1;
        point.y = rand() / (float) RAND_MAX * 2 - 1;
        point.z = rand() / (float) RAND_MAX * 2 - 1;
        float distSqr = vecDot(point, point);
        if (distSqr > 1)
            continue;

        velocity.x = rand() / (float) RAND_MAX * 2 - 1;
        velocity.y = rand() / (float) RAND_MAX * 2 - 1;
        velocity.z = rand() / (float) RAND_MAX * 2 - 1;\
        distSqr = vecDot(velocity, velocity);
        if (distSqr > 1)
            continue;

        positions[posIdx++] = point.x * scale;
        positions[posIdx++] = point.y * scale;
        positions[posIdx++] = point.z * scale;
        positions[posIdx++] = 1.0f;

        velocities[velPtr++] = velocity.x * velScale;
        velocities[velPtr++] = velocity.y * velScale;
        velocities[velPtr++] = velocity.z * velScale;
        velocities[velPtr++] = 1.0f; 

        i++;
    }
}

float3CPU CPUmethods::scaleVector(float3CPU &vec, float scalar)
{
    float3CPU rt = vec;
    rt.x *= scalar;
    rt.y *= scalar;
    rt.z *= scalar;

    return rt;
}

float CPUmethods::normalize(float3CPU &vec)
{
    float norm = sqrtf(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
    if (norm > 1e-6){
        vec.x /= norm;
        vec.y /= norm;
        vec.z /= norm;
    }
    return norm;
}

float CPUmethods::vecDot(float3CPU vec1, float3CPU vec2)
{
    return vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z;
}

float3CPU CPUmethods::crossDot(float3CPU vec1, float3CPU vec2)
{
    float3CPU rt;
    rt.x *= vec1.y*vec2.z - vec1.z*vec2.y;
    rt.y *= vec1.z*vec2.x - vec1.x*vec2.z;
    rt.z *= vec1.x*vec2.y - vec1.y*vec2.x;
    return rt;
}

NbodySimulationCPU::NbodySimulationCPU(int numBodies) : NBodySimulation(numBodies)
{
    
    memRead = 0;
    memWrite = 1;
    // timer = 0;

    // mem_Position[0] = mem_Position[1] = 0;
    // mem_Velocity[0] = mem_Velocity[1] = 0;

    _initialize(numBodies);
    setPhysicalParams(0.1f, 0.995f);
}

NbodySimulationCPU::~NbodySimulationCPU()
{
    _summarize();
    nBodies = 0;
}

void NbodySimulationCPU::run(float dt)
{
    // devTimer.startTimer();
    simulate(dt);
    // devTimer.stopTimer();

    std::swap(memRead, memWrite);
}

void NbodySimulationCPU::setPhysicalParams(float damping, float e2)
{
    m_damping = damping;
    m_e2 = e2;
}

float *NbodySimulationCPU::readMemory(PhysicalQuantity arr)
{
    float *data = 0;
    switch(arr)
    {
        case POSITION:
            data = mem_Position;
            break;
        case VELOCITY:
            data = mem_Velocity;
            break;
    }
    return data;
}

void NbodySimulationCPU::writeMemory(PhysicalQuantity arr, float *data)
{
    float *inData = 0;

    switch(arr)
    {
        case POSITION:
            inData = mem_Position;
            break;
        case VELOCITY:
            inData = mem_Velocity;
            break;
    }
    
    memcpy(inData, data, 4*nBodies*sizeof(float));
}

void NbodySimulationCPU::_initialize(int numBodies)
{
    mem_Position = new float[4*nBodies];
    mem_Velocity = new float[4*nBodies];
    mem_Force = new float[3*nBodies];

    memset(mem_Position, 0, 4*nBodies*sizeof(float));
    memset(mem_Velocity, 0, 4*nBodies*sizeof(float));
    memset(mem_Force, 0, 3*nBodies*sizeof(float));

    // devTimer.createTimer();
}

void NbodySimulationCPU::_summarize()
{
    delete [] mem_Position;
    delete [] mem_Velocity;
    delete [] mem_Force;
    // devTimer.resetTimer();
}

void NbodySimulationCPU::simulate(float dt)
{
    computeForce();
    #ifdef OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < nBodies; ++i)
    {
        int idx = 4*i;
        int indexForce = 3 * i;
        float position[3], velocity[3], force[3];

        position[0] = mem_Position[idx+0];
        position[1] = mem_Position[idx+1];
        position[2] = mem_Position[idx+2];
        float massInv = mem_Position[idx+3];
        
        velocity[0] = mem_Velocity[idx+0];
        velocity[1] = mem_Velocity[idx+1];
        velocity[2] = mem_Velocity[idx+2];
        // float massInv = mem_Velocity[idx+3];

        force[0] = mem_Force[idx+0];
        force[1] = mem_Force[idx+1];
        force[2] = mem_Force[idx+2];

        velocity[0] += (force[0] * massInv) * dt; 
        velocity[1] += (force[1] * massInv) * dt; 
        velocity[2] += (force[2] * massInv) * dt;

        velocity[0] *= m_damping; 
        velocity[1] *= m_damping; 
        velocity[2] *= m_damping;

        position[0] += velocity[0] * dt;
        position[1] += velocity[1] * dt;
        position[2] += velocity[2] * dt;

        mem_Position[idx+0] = position[0];
        mem_Position[idx+1] = position[1];
        mem_Position[idx+2] = position[2];

        mem_Velocity[idx+0] = velocity[0];
        mem_Velocity[idx+1] = velocity[1];
        mem_Velocity[idx+2] = velocity[2];
        // mem_Velocity[idx+3] = massInv;   
    }
}

void NbodySimulationCPU::computeForce()
#ifdef OPENMP
#pragma omp parallel for
#endif
{
    for (int i = 0; i < nBodies; i++) 
    {
        int idxForce = 3 * i;
        float acceleration[3] = {0, 0, 0};
        // We unroll this loop 4X for a small performance boost.
        int j = 0;
        while (j < nBodies) 
        {
            interact(acceleration, &mem_Position[4 * i], &mem_Position[4 * j], m_e2); j++;
            interact(acceleration, &mem_Position[4 * i], &mem_Position[4 * j], m_e2); j++;
            interact(acceleration, &mem_Position[4 * i], &mem_Position[4 * j], m_e2); j++;
            interact(acceleration, &mem_Position[4 * i], &mem_Position[4 * j], m_e2); j++;
        }

        mem_Force[idxForce] = acceleration[0];
        mem_Force[idxForce + 1] = acceleration[1];
        mem_Force[idxForce + 2] = acceleration[2];
    }
}

// void NbodySimulationCPU::computeForce()
// {
//     for (int i = 0; i < nBodies; ++i)
//     {
//         mem_Force[4*i] = mem_Force[4*i + 1] = mem_Force[4*i + 2] = 0;
        
//         for (int j = 0; j < nBodies; ++j)
//         {
//             float acceleration[3] = {0,0,0};
//             interact(acceleration, &mem_Position[memRead][4*i], &mem_Position[memRead][4*j], m_e2);
            
//             for (int k = 0; k < 3; ++k)
//             {
//                 mem_Force[4*i + k] = acceleration[k];
//             }
//         }
//     }  
// }

void NbodySimulationCPU::interact(float acceleration[3], float position1[4], 
                                  float position2[4], float e2)
{
    float r[3];

    r[0] = position2[0] - position1[0];
    r[1] = position2[1] - position1[1];
    r[2] = position2[2] - position1[2];

    float distSqr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    distSqr += e2;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float invDist = (float)1.0 / (float)sqrt((double)distSqr);
    float invDistCube = invDist * invDist * invDist;
    float s = position2[3] * invDistCube;

    acceleration[0] += r[0] * s;
    acceleration[1] += r[1] * s;
    acceleration[2] += r[2] * s;

}
