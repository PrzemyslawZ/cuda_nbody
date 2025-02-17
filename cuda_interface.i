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

// %inline %{
//     extern void simulate(
//         float *inputBuffer, 
//         float *resultsBuffer,  
//         std::map<std::string, float> params, 
//         std::map<std::string, int> gpuParams);

//         extern double TIME_GPU, TIME_CPU;
//         extern std::string PLATFORM;


// %}

// %init %{
// import_array();
// %}
%array_class(float, floatArray);

%include "cuda_interface.h"


// extern double TIME_GPU, TIME_CPU;
// extern std::string PLATFORM;

extern void simulate(
    float *inputBuffer, 
    float *resultsBuffer, 
    std::map<std::string, float> params, 
    std::map<std::string, int> gpuParams);

// %apply (float* IN_ARRAY1, int DIM1) {(float* inputBuffer, int ibuffDim)};
// %apply (float* IN_ARRAY1, int DIM1) {(float* resultsBuffer, int rbuffDim)};
