%{
#define SWIG_FILE_WITH_INIT
#include "cuda_interface.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%inline %{
extern double TIME_GPU, TIME_CPU;
extern std::string PLATFORM;
%}

void simulate(float *resultsBuffer,
              const std::map<std::string, float> params, 
              const std::map<std::string, int> gpuParams)

%apply (float* IN_ARRAY1, int DIM1) {(float* resultsBuffer, int buffDim)};
%include "cuda_interface.h"
