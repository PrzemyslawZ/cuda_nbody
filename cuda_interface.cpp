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


void runGPUSimulation()
{
    timer->resetTimer();
    timer->startTimer();
    for (int tStep = 0; tStep < PARAMS.steps; tStep++)
    {
        kernel->run(tStep);
    }    
    cudaDeviceSynchronize();
    timer->stopTimer();
    TIME_GPU = timer->elapsedTime;
}


void runCPUSimulation()
{
    timer->resetTimer();
    timer->startTimer();
    for (int tStep = 0; tStep < PARAMS.steps; tStep++)
    {
        host->run(tStep);
    }
    timer->stopTimer();
    TIME_CPU = timer->elapsedTime;
}


void initializeSimulator()
{
    if(PLATFORM == "GPU" || "BOTH")
        kernel = new KernelHandler(PARAMS, numBlocks, blockSize);
    if(PLATFORM == "CPU")
        host = new NbodySimulationCPU(PARAMS)
}


void loadParams(const std::map<std::string, float> params, 
                const std::map<std::string, int> gpuParams)
{
    PARAMS.systemType = params["system_type"];
    PARAMS.numBodies = (int)params["num_bodies"]
    PARAMS.dt = params["dt"];
    PARAMS.steps = (int)params["tsteps"];
    //PARAMS.savesteps = 100;
    PARAMS.Nx_spins = (int)params["nx_spins"];
    PARAMS.Ny_spins = (int)params["ny_spins"];
    PARAMS.Nz_spins = (int)params["ny_spins"];
    PARAMS.Gamma1 = params["gamma1"];
    PARAMS.Gamma2 = params["gamma1"];
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


void initialize(const std::map<std::string, float> params, 
                const std::map<std::string, int> gpuParams)
{
    loadParams(params, gpuParams)
    initializeSimulator();
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


void summarize(float *resultsBuffer, int buffDim)
{
    if(kernel)
        resultsBuffer[buffDim] = kernel->readMemory();                
    if(host)
        resultsBuffer[2*buffDim] = host->readMemory();                    
    closeSimulation();
}


void simulate(float *resultsBuffer, int buffDim,  
              const std::map<std::string, float> params, 
              const std::map<std::string, int> gpuParams)
{
    initialize(params, gpuParams);
    if(kernel)
        runGPUSimulation();
    if(host)
        runCPUSimulation();
        
    summarize(buffDim);
};

// TODO: Implement reading results from gpu
// TODO: Implement arrays numpy for getting reslts 
