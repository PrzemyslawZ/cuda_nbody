###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_INSTALL_PATH=/usr/local/cuda

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=g++
CC_FLAGS= --std=c++11 -m64
CC_LIBS=

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc -ccbin $(CC) -use_fast_math -gencode arch=compute_86,code=sm_86
NVCC_FLAGS=
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_INSTALL_PATH)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_INSTALL_PATH)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart -lcuda

##########################################################
# nvcc simulation.cu -m64 -ccbin g++ --std=c++11 -gencode arch=compute_86,code=sm_86 -o simulation
## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

##########################################################

## Make variables ##

# Target executable name:
EXE = nbodyBench
DATA_GPU = DATA_perfGPU
DATA_CPU = DATA_perfCPU

$(shell mkdir -p $(OBJ_DIR))

# Object files:
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/kernel_handler.o $(OBJ_DIR)/nbodyCPU.o $(OBJ_DIR)/device_timer.o\
		$(OBJ_DIR)/kernel.o #$(OBJ_DIR)/nbody_KERNEL.o
#$(OBJ_DIR)/device_timer.o $(OBJ_DIR)/kernel_handler.o $(OBJ_DIR)/nbodyCPU.o 
##########################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp include/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)
	$(RM) *.txt
	$(RM) *.txt