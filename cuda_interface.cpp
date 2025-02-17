#include "./cuda_interface.h"
#include "./include/kernel_handler.h"
#include "./include/nbodyCPU.h"
#include "./include/device_timer.h"

KernelHandler *kernel = 0;
NbodySimulationCPU *host = 0;
DeviceTimer *timer = 0;

PhysicalParams *PARAMS = 0;
GPUDev *GPU_PARAMS = 0;

double TIME_GPU = 0;
double TIME_CPU = 0;
std::string PLATFORM = "CPU";


void loadParams(std::map<std::string, float> params, 
                std::map<std::string, int> gpuParams)
{
    PARAMS = new PhysicalParams;
    GPU_PARAMS = new GPUDev;

    PARAMS->systemType = params["system_type"];
    PARAMS->numBodies = (int)params["num_bodies"];
    PARAMS->dt = params["dt"];
    PARAMS->steps = (int)params["tsteps"];
    PARAMS->saveStep = (int)params["save_step"];
    PARAMS->Nx_spins = (int)params["nx_spins"];
    PARAMS->Ny_spins = (int)params["ny_spins"];
    PARAMS->Nz_spins = (int)params["ny_spins"];
    PARAMS->Gamma1 = params["gamma1"];
    PARAMS->Gamma2 = params["gamma2"];
    PARAMS->Nth1 = (int)params["nth1"];
    PARAMS->Nth2 = (int)params["nth2"];
    PARAMS->Gamma = params["gamma"];
    PARAMS->GammaPhase = params["gamma_phase"];
    PARAMS->Jx = params["jx"];
    PARAMS->Jy = params["jy"];
    PARAMS->Jz = params["jz"];
    PARAMS->Omegax = params["Omega_x"];
    PARAMS->Omegay = params["Omega_y"];
    PARAMS->Omegaz = params["Omega_z"];
    PARAMS->OmegaD = params["Omega_D"];  
    //PARAMS->startMeasuring = params["Omega_x"];
    //strcpy(PARAMS->filename, "results.txt");
    //strcpy(PARAMS->directory, "./");
    //PARAMS->ThreadId = 0;   
    GPU_PARAMS->blockSize = gpuParams["block_size"];
    GPU_PARAMS->numBlocks = gpuParams["blocks_num"];
}


void initializeSimulator(float *inputBuffer)
{
    if(PLATFORM == "GPU" || "BOTH")
    {
        kernel = new KernelHandler(PARAMS, GPU_PARAMS->numBlocks, GPU_PARAMS->blockSize);
        kernel->writeMemory(inputBuffer);
    }
    if(PLATFORM == "CPU" || "BOTH")
    {
        host = new NbodySimulationCPU(PARAMS);
        host->writeMemory(inputBuffer);
    }
}


void initialize(float *inputBuffer,
                std::map<std::string, float> params, 
                std::map<std::string, int> gpuParams)
{
    loadParams(params, gpuParams);
    initializeSimulator(inputBuffer);
}


void runGPUSimulation(float *resultsBuffer)
{
    int ptrShift = 2 * PARAMS->numBodies;
    int iSave = 0;

    timer->resetTimer();
    timer->startTimer();
    for (int tStep = 0; tStep < PARAMS->steps; tStep++)
    {
        kernel->run(tStep);
        if(tStep%PARAMS->saveStep==0)
        {
            memcpy(resultsBuffer + ptrShift*(iSave++), 
                   kernel->readMemory(), 
                   ptrShift * sizeof(float));
        }
    }    
    cudaDeviceSynchronize();
    timer->stopTimer();
    TIME_GPU = timer->elapsedTime;
}


void runCPUSimulation(float *resultsBuffer)
{
    int ptrShift = 2 * PARAMS->numBodies;
    int iSave = 0;

    timer->resetTimer();
    timer->startTimer();
    for (int tStep = 0; tStep < PARAMS->steps; tStep++)
    {
        host->run(tStep);
        if(tStep%PARAMS->saveStep==0)
        {
            memcpy(resultsBuffer + ptrShift*(iSave++), 
                   host->readMemory(), 
                   ptrShift * sizeof(float));
        }
    }
    timer->stopTimer();
    TIME_CPU = timer->elapsedTime;
}


void closeSimulation()
{
    if(kernel)
        delete kernel;
    if(host)
        delete host;
    if(timer)
        delete timer;
    if(PARAMS)
        delete PARAMS;
    if(GPU_PARAMS)
        delete GPU_PARAMS;
}


// void summarize(float *resultsBuffer, int buffDim)
// {
//     if(kernel)
//         resultsBuffer[buffDim] = kernel->readMemory();                
//     if(host)
//         resultsBuffer[2*buffDim] = host->readMemory();                    
//     closeSimulation();
// }

void simulate(
    float *inputBuffer, 
    float *resultsBuffer, 
    std::map<std::string, float> params, 
    std::map<std::string, int> gpuParams)
{
    // initialize(inputBuffer, params, gpuParams);
    // if(kernel)
    //     runGPUSimulation(resultsBuffer);
    // if(host)
    //     runCPUSimulation(resultsBuffer);

    resultsBuffer[0] = 1;
    resultsBuffer[1] = 10;
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("Device Number: %d\n", i);
      printf("  Device name: %s\n", prop.name);
      printf("  Memory Clock Rate (KHz): %d\n",
             prop.memoryClockRate);
      printf("  Memory Bus Width (bits): %d\n",
             prop.memoryBusWidth);
      printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
             2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
    // summarize(buffDim);
}; 


