#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cmath>
#include <random>
#include <unistd.h> 

#include "./include/kernel_handler.h"
#include "./include/nbodyCPU.h"
#include "./include/device_timer.h"


KernelHandler *kernel = 0;
NbodySimulationCPU *nCPU = 0;
CPUmethods *cpuMethods = 0;
DeviceTimer *timer = 0;

int blockSize = 4;
int numBlocks = 1;

float ERR = 0.01;
float dt = 0.0001f;
float* resultsBuffer = 0;

double TIME_GPU, TIME_CPU;

FILE *f;
FILE *outputFileCPU;
FILE *outputFileGPU;
std::string filename = "DATA";


void randomizeSystem(float *positions, int numBodies)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> rand01(0, 1);  // Generates 0 or 1

    int posIdx = 0, i = 0;
    while (i < numBodies)
    {
        float2 point;
        point.x = rand() / (float) RAND_MAX * 2 * 3.141;
        point.y = rand() / (float) RAND_MAX * 2 * 3.141;

        int randomSign = 1 - 2 * rand01(gen);  // Gives -1 or +1
        int randomBit = rand01(gen);  // Generates 0 or 1
        
        positions[posIdx++] = point.x;
        positions[posIdx++] = point.y;

        i++;
    }
}

void createBuffers(int numBodies)
{
    resultsBuffer = new float[numBodies*2];
    randomizeSystem(positions, numBodies);
}

void loadBuffer(NBodySimulation *caller)
{
    caller->writeMemory(NBodySimulation::POSITION, resultsBuffer); // no need for enum 
}

void compareBuffers(int numBodies)
{
    float *posCPU = nCPU->readMemory(NBodySimulation::POSITION); // no need for enum 
    float *posGPU = kernel->readMemory(NBodySimulation::POSITION); // no need for enum 

    int result = 0;
    for (int i = 0; i < numBodies; i++)
    {
        if(abs(posGPU[i]-posCPU[i]) < ERR)
            result +=1;
        //else
        //    printf("\n%d GPU_pos=%f CPU_pos=%f INIT_pos=%f\n",i, posGPU[i], posCPU[i], positions[i]);
        //printf("%d GPU_vel=%f CPU_vel=%f INIT_vel=%f\n",i, velGPU[i], velCPU[i], velocities[i]);
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

void runGPUSimulation(int numBodies, ThreadInput input)
{
//    kernel->run(dt);
    float sqr3 = sqrt(3);
    timer->resetTimer();
    timer->startTimer();
    outputFileGPU = fopen("resultsGPU.txt", "w");
    for (int step = 0; step < input.steps; step++)
    {
        kernel->run(step);
        if(step%10==0){
            float *posGPU = kernel->readMemory(NBodySimulation::POSITION);
            for (int i = 0; i < numBodies; i++){
                fprintf(outputFileGPU, "%f, %f, %f \n", sqr3 * sin(posGPU[i*2]) * sin(posGPU[i*2+1]), -sqr3 * sin(posGPU[i*2]) * cos(posGPU[i*2+1]), -sqr3 * cos(posGPU[i*2]));
                //fprintf(outputFile, "%f, %f \n", posGPU[i*2], posGPU[i*2+1]);
            }
        }
    }    
    cudaDeviceSynchronize();
    timer->stopTimer();
    TIME_GPU = timer->elapsedTime;
    summarizeTest(numBodies, input.steps, GFLOPS_GPU);
    printf("GPU performance for N=%d gflops/s=%0.3f  time=%0.3f ms\n", numBodies, GFLOPS_GPU, TIME_GPU);
}

void runCPUSimulation(int numBodies, ThreadInput input)
{
    timer->resetTimer();
    timer->startTimer();
    float sqr3 = sqrt(3);
    outputFileCPU = fopen("resultsCPU.txt", "w");

    for (int step = 0; step < input.steps; step++)
    {
        nCPU->run(step);
        if(step%10==0){
            float *posCPU = nCPU->readMemory(NBodySimulation::POSITION);
            for (int i = 0; i < numBodies; i++){
                fprintf(outputFileCPU, "%f, %f, %f \n", sqr3 * sin(posCPU[i*2]) * sin(posCPU[i*2+1]), -sqr3 * sin(posCPU[i*2]) * cos(posCPU[i*2+1]), -sqr3 * cos(posCPU[i*2]));
                //fprintf(outputFile, "%f, %f \n", posGPU[i*2], posGPU[i*2+1]);
            }
        }
    }
    timer->stopTimer();
    TIME_CPU = timer->elapsedTime;
    summarizeTest(numBodies, input.steps, GFLOPS_CPU);
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
}

void benchmark(int bodies_repeats, ThreadInput input){

    int particles = 0;
    for (int i = 0; i < bodies_repeats; i++)
    {
        particles = pow(2,i) * input.Nx_spins * input.Ny_spins * input.Nz_spins;
        kernel = new KernelHandler(particles, numBlocks, blockSize, input);
        nCPU = new NbodySimulationCPU(particles, input);

        printf("\n##################################### TEST %d. #####################################'\n", i);
        createBuffers(particles);
        initialize(particles, "gpu");
        runGPUSimulation(particles, input);

        initialize(particles, "cpu");
        runCPUSimulation(particles, input);

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
    int bodies_repeats = 1;

    cpuMethods = new CPUmethods();
    timer = new DeviceTimer();

    ThreadInput input;
    input.systemType = 16;
    input.dt = 0.001;
    input.steps = 1000000;
    input.savesteps = 100;
    input.Nx_spins = 64;
    input.Ny_spins = 64;
    input.Nz_spins = 1;
    input.Gamma1 = 0;
    input.Gamma2 = 0;
    input.Nth1 = 0;
    input.Nth2 = 0;
    input.Gamma = 1;
    input.GammaPhase = 0;
    input.Jx = 0.5;
    input.Jy = 0.2;
    input.Jz = 0.7;
    input.Omegax = 0;
    input.Omegay = 0;
    input.Omegaz = 0;
    input.OmegaD = 0;  
    input.startMeasuring = 0;
    strcpy(input.filename, "results.txt");
    strcpy(input.directory, "./");
    input.ThreadId = 0;

    saveData();
    benchmark(bodies_repeats, input);

    delete cpuMethods;
    delete timer;
    fclose(f);
    fclose(outputFileCPU);
    fclose(outputFileGPU);

    printf("\nfinished");
    return 0;

}
