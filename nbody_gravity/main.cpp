#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "./include/kernel_handler.h"
#include "./include/nbodyCPU.h"
#include "./include/device_timer.h"


KernelHandler *kernel = 0;
NbodySimulationCPU *nCPU = 0;
CPUmethods *cpuMethods = 0;
DeviceTimer *timer = 0;

int blockSize = 4;
int numBlocks = 1;

float ERR = 0.001;
float dt = 0.001f;
float* positions = 0;
float* velocities = 0;
double GFLOPS_GPU; 
double GFLOPS_CPU;
double TIME_GPU, TIME_CPU;

FILE *f;
std::string filename = "DATA";

void createBuffers(int numBodies)
{
    positions = new float[numBodies*4];
    velocities = new float[numBodies*4];

    cpuMethods->randomizeSystem(positions, velocities,
            1.52f, 2.f, numBodies);
}

void loadBuffer(NBodySimulation *caller)
{
    caller->writeMemory(NBodySimulation::POSITION, positions);
    caller->writeMemory(NBodySimulation::VELOCITY, velocities);
}
void compareBuffers(int numBodies)
{
    float *posCPU = nCPU->readMemory(NBodySimulation::POSITION);
    float *posGPU = kernel->readMemory(NBodySimulation::POSITION);

    float *velCPU = nCPU->readMemory(NBodySimulation::VELOCITY);
    float *velGPU = kernel->readMemory(NBodySimulation::VELOCITY);

    int result = 0;
    for (int i = 0; i < numBodies; i++)
    {
        if(abs(posGPU[i]-posCPU[i]) < ERR)
            result +=1;
        // else
        //     printf("\n%d GPU_pos=%f CPU_pos=%f INIT_pos=%f\n",i, posGPU[i], posCPU[i], positions[i]);
        // printf("%d GPU_vel=%f CPU_vel=%f INIT_vel=%f\n",i, velGPU[i], velCPU[i], velocities[i]);
    }
    printf("GPU vs CPU results: %0.2f %\n", (float)(result / numBodies) * 100.0);
}

void initialize(int numBodies, std::string test){
    if(test=="gpu")
        loadBuffer(kernel);
    if(test=="cpu")
        loadBuffer(nCPU);
    timer->createTimer();
}

void summarizeTest(int numBodies, int repeats, double &glfops){
    const int flopsPerInteraction = 20;
    double interactionsPerSecond = (float)numBodies * (float)numBodies;
    interactionsPerSecond *= 1e-9 * repeats * 1000 / timer->elapsedTime;
    glfops = interactionsPerSecond * (float)flopsPerInteraction;
}

void runGPUSimulation(int repeats, int numBodies)
{
    kernel->run(dt);
    timer->resetTimer();    // svaes some time in the measuremennt
    timer->startTimer();
    for (int i = 0; i < repeats; i++)
    {
        kernel->run(dt);
    }    
    cudaDeviceSynchronize();
    timer->stopTimer();
    TIME_GPU = timer->elapsedTime;
    summarizeTest(numBodies, repeats, GFLOPS_GPU);
    printf("GPU performance for N=%d gflops/s=%0.3f  time=%0.3f ms\n", numBodies, GFLOPS_GPU, TIME_GPU);
}

void runCPUSimulation(int repeats, int numBodies)
{
    timer->resetTimer();
    timer->startTimer();
    for (int i = 0; i < repeats; i++)
    {
        nCPU->run(dt);
    }
    timer->stopTimer();
    TIME_CPU = timer->elapsedTime;
    summarizeTest(numBodies, repeats, GFLOPS_CPU);
    printf("CPU performance for N=%d gflops/s=%0.3f  time=%0.3f ms\n", numBodies, GFLOPS_CPU, TIME_CPU);
}

void saveData(){
    
    std::string filename = "./DATA_nBlocks_" +\
     std::to_string(numBlocks)+ "_blockSize_" +\
      std::to_string(blockSize) + ".txt"; 
    f = fopen(filename.c_str(), "a");
    fprintf(f, "N_BODY\tGFLOPS_GPU\tGFLOPS_CPU\tTIME_GPU\tTIME_CPU\tGPUvsCPU\n");
}

void endBenchmark(){
    if(kernel)
        delete kernel;
    if(nCPU)
        delete nCPU;
    free(positions);
    free(velocities);
}

void benchmark(int bodies_repeats, int repeats, int numBodies){

    int particles = 0;
    for (int i = 1; i < bodies_repeats; i++)
    {
        particles = i* numBodies;
        kernel = new KernelHandler(particles, numBlocks, blockSize);
        nCPU = new NbodySimulationCPU(particles);

        printf("\n##################################### TEST %d. #####################################'\n", i);
        createBuffers(particles);
        initialize(particles, "gpu");
        runGPUSimulation(repeats, particles);

        initialize(particles, "cpu");
        runCPUSimulation(repeats, particles);

        compareBuffers(particles);
        printf("CPU vs GPU performance: gflops ratio=%0.3f  time ratio=%0.3f\n", 
            GFLOPS_GPU/GFLOPS_CPU, 
            TIME_CPU/TIME_GPU);
        fprintf(f, "%d\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\n", 
            particles,
             GFLOPS_GPU, 
             GFLOPS_CPU, 
             TIME_GPU, 
             TIME_CPU ,
             GFLOPS_GPU/GFLOPS_CPU);
        endBenchmark();
    }
}

int main(int argc, char const *argv[])
{
    cudaDeviceSynchronize();
    int bodies_repeats = 10;
    int repeats = 100;
    int numBodies = 128;

    cpuMethods = new CPUmethods();
    timer = new DeviceTimer();

    saveData();
    benchmark(bodies_repeats, repeats, numBodies);

    delete cpuMethods;
    delete timer;
    fclose(f);
    return 0;

}
