%{
#define SWIG_FILE_WITH_INIT
#include "cuda_interface.h"
%}

%include "numpy.i"
%include "std_string.i"
%include "std_map.i"

namespace std {
    %template(map_string_float) map<string, float>;
    %template(map_string_int) map<string, int>;
}

%init %{
import_array();
%}

%inline %{
extern double TIME_GPU, TIME_CPU;
extern std::string PLATFORM;
%}

void simulate(float *inputBuffer, int ibuffDim, 
              float *resultsBuffer, int rbuffDim,  
              const std::map<std::string, float> params, 
              const std::map<std::string, int> gpuParams)

%apply (float* IN_ARRAY1, int DIM1) {(float* inputBuffer, int ibuffDim)};
%apply (float* IN_ARRAY1, int DIM1) {(float* resultsBuffer, int rbuffDim)};
%include "cuda_interface.h"
