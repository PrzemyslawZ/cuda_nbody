## How to start

1. To use the CUDA code you need: GPU, nvcc compiler and SWIG tool, where all of those tools are available on CIP computers
2. To build C++ code just type: "make" in a console
3. To build Python wrapped version of code type: "make swig"
4. In case you want to wipe out all files created during compilation run: "make clean"

## How to use

In order to communicate with C code via Python you need to files: main.py (where you execute code and process data, BTW you are welcome to rename it as you want) and pycu_spins.py which is the simple interface to prepare and communicate with library. 
All others files generated in process of compilation are created automatically. Files like cuda_interface.* are necessary for SWIG tool and should have certain structure.

## Some tipps 

To improve / track the performance of GPU (currently only on cip2ryzen4) run in console nvtop to see online GPU monitor.