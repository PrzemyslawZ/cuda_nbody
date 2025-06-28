%module cuda_interface

%{
#define SWIG_FILE_WITH_INIT
#include "cuda_interface.h"
%}

%include "carrays.i"
%include "std_string.i"
%include "std_map.i"


namespace std {
    %template(map_string_float) map<string, float>;
    %template(map_string_int) map<string, int>;
}

%array_class(float, floatArray);
%include "cuda_interface.h"


extern void simulate(
    float *inputBuffer, 
    float *resultsBuffer, 
    std::map<std::string, float> params, 
    std::map<std::string, int> gpuParams);


// IN CASE OF NUMPY ARRAYS 
// %init %{
// import_array();
// %}
// %apply (float* IN_ARRAY1, int DIM1) {(float* inputBuffer, int ibuffDim)};
// %apply (float* IN_ARRAY1, int DIM1) {(float* resultsBuffer, int rbuffDim)};
