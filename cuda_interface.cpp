#include "./cuda_interface.h"

#define BLOCK_SIZE_DEF = 256;
#define NUM_BLOCKS_DEF = 4;


struct GPUDev{

    int blocksSize = BLOCK_SIZE_DEF;
    int numBlocks = NUM_BLOCKS_DEF;
}


KernelHandler *kernel = 0;
NbodySimulationCPU *host = 0;
DeviceTimer *timer = 0;
PhysicalParams PARAMS; 
GPUDev GPU_PARAMS;


void runGPUSimulation(float *resultsBuffer)
{
    int ptrShift = 2 * PARAMS.numBodies;
    int iSave = 0;

    timer->resetTimer();
    timer->startTimer();
    for (int tStep = 0; tStep < PARAMS.steps; tStep++)
    {
        kernel->run(tStep);
        if(tStep%PARAMS.saveStep==0)
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
    int ptrShift = 2 * PARAMS.numBodies;
    int iSave = 0;

    timer->resetTimer();
    timer->startTimer();
    for (int tStep = 0; tStep < PARAMS.steps; tStep++)
    {
        host->run(tStep);
        if(tStep%PARAMS.saveStep==0)
        {
            memcpy(resultsBuffer + ptrShift*(iSave++), 
                   host->readMemory(), 
                   ptrShift * sizeof(float));
        }
    }
    timer->stopTimer();
    TIME_CPU = timer->elapsedTime;
}


void initializeSimulator(float *inputBuffer)
{
    if(PLATFORM == "GPU" || "BOTH")
    {
        kernel = new KernelHandler(PARAMS, numBlocks, blockSize);
        kernel->writeMemory(inputBuffer)
    }
    if(PLATFORM == "CPU" || "BOTH")
    {
        host = new NbodySimulationCPU(PARAMS)
        host->writeMemory(inputBuffer)
    }
}


void loadParams(const std::map<std::string, float> params, 
                const std::map<std::string, int> gpuParams)
{
    PARAMS.systemType = params["system_type"];
    PARAMS.numBodies = (int)params["num_bodies"]
    PARAMS.dt = params["dt"];
    PARAMS.steps = (int)params["tsteps"];
    PARAMS.saveStep = (int)params["save_step"];
    PARAMS.Nx_spins = (int)params["nx_spins"];
    PARAMS.Ny_spins = (int)params["ny_spins"];
    PARAMS.Nz_spins = (int)params["ny_spins"];
    PARAMS.Gamma1 = params["gamma1"];
    PARAMS.Gamma2 = params["gamma2"];
    PARAMS.Nth1 = (int)params["nth1"];
    PARAMS.Nth2 = (int)params["nth2"];
    PARAMS.Gamma = params["gamma"];
    PARAMS.GammaPhase = params["gamma_phase"];
    PARAMS.Jx = params["jx"];
    PARAMS.Jy = params["jy"];
    PARAMS.Jz = params["jz"];
    PARAMS.Omegax = params["Omega_x"];
    PARAMS.Omegay = params["Omega_y"];
    PARAMS.Omegaz = params["Omega_z"];
    PARAMS.OmegaD = params["Omega_D"];  
    //PARAMS.startMeasuring = params["Omega_x"];
    //strcpy(PARAMS.filename, "results.txt");
    //strcpy(PARAMS.directory, "./");
    //PARAMS.ThreadId = 0;   
    GPU_PARAMS.blockSize = gpuParams["block_size"];
    GPU_PARAMS.numBlocks = gpuParams["blocks_num"];
}


void initialize(float *inputBuffer,
                const std::map<std::string, float> params, 
                const std::map<std::string, int> gpuParams)
{
    loadParams(params, gpuParams)
    initializeSimulator(inputBuffer);
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

/*
void summarize(float *resultsBuffer, int buffDim)
{
    if(kernel)
        resultsBuffer[buffDim] = kernel->readMemory();                
    if(host)
        resultsBuffer[2*buffDim] = host->readMemory();                    
    closeSimulation();
}

*/


void simulate(float *inputBuffer, int ibuffDim, 
              float *resultsBuffer, int rbuffDim,  
              const std::map<std::string, float> params, 
              const std::map<std::string, int> gpuParams)
{
    initialize(inputBuffer, params, gpuParams);
    if(kernel)
        runGPUSimulation(resultsBuffer);
    if(host)
        runCPUSimulation();
        
    summarize(buffDim);
}; 
