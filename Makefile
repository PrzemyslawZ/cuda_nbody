
## CUDA directory ##
############################################################
CUDA_INSTALL_PATH:=/usr/local/cuda
BUILD_TYPE := release

## PYTHON DIR ##
############################################################
PYTHON_INSTALL_PATH := -I/usr/include/python3.12


## CC COMPILER OPTIONS ##
############################################################
ALL_CCFLAGS :=
CC:=g++
CC_FLAGS:= --std=c++11 -fPIC -shared
CC_LIBS:=

## SWIG WRAPPER OPTION ##
############################################################
SWIG := swig
SWIGFLAGS := -c++ -python
SWIG_INCL := $(PYTHON_INSTALL_PATH)

## NVCC COMPILER OPTIONS ##
############################################################
NVCC := $(CUDA_INSTALL_PATH)/bin/nvcc -ccbin $(CC)
NVCC_FLAGS:= -m64 -use_fast_math
NVCC_LIBS:=
GENCODE_FLAGS := -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86

ALL_CCFLAGS += $(NVCC_FLAGS) 
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

## SOURCES ##
##########################################################
SRC_DIR = src
OBJ_DIR = bin
INC_DIR = include

INCLUDES  := -I$(CUDA_INSTALL_PATH)/include -I$(INC_DIR)/include 
LIBRARIES := -L$(CUDA_INSTALL_PATH)/lib64
CUDA_LINK_LIBS:= -lcudart -lcuda #lpthread -lcublas

## COMPILE ##
##########################################################
SAMPLE_ENABLED := 1
ifeq ($(SAMPLE_ENABLED),0)
	EXEC ?= @echo "[@]"
endif

all: build

build: nbodyBench

kernel.o: $(SRC_DIR)/kernel.cu 
	$(EXEC) $(NVCC)  --compiler-options -fPIC -shared $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $< 

cuda_interface.o: cuda_interface.cpp
	$(EXEC) $(CC) $(INCLUDES) $(SWIG_INCL) $(CC_FLAGS) -o $@ -c $<

_cuda_interface.so: cuda_interface.o cuda_interface_wrap.o kernel.o
	$(EXEC) $(CC) $(CC_FLAGS) -o $@  $^ -lcudart
#-l cudart

cuda_interface_wrap.cxx: cuda_interface.i
	$(SWIG) $(SWIGFLAGS) $<

cuda_interface_wrap.o: cuda_interface_wrap.cxx
	$(EXEC) $(CC) $(INCLUDES) $(SWIG_INCL) $(CC_FLAGS) -o $@ -c $<

main.o: main.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

nbodyBench: main.o kernel.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES) $(CUDA_LINK_LIBS)
	$(EXEC) mkdir -p $(OBJ_DIR)

run: build
	$(EXEC) ./nbodyBench

swig: _cuda_interface.so

clean:
	$(RM) bin/* *.o
	$(RM)  *.o
	$(RM) ./nbodyBench
	$(RM) ./_cuda_interface.py
	$(RM) ./cuda_interface.py
	$(RM)  *.o
	$(RM) ./cuda_interface_wrap.cxx
	$(RM) ./_cuda_interfac.so


# /usr/local/cuda/bin/nvcc -ccbin g++ -I../../../Common -m64 -ftz=true --threads 0 --std=c++11 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -o bodysystemcuda.o -c bodysystemcuda.cu

# /usr/local/cuda/bin/nvcc -ccbin g++ -I../../../Common 
# -m64 -ftz=true --threads 0 --std=c++11  -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -o nbody.o -c nbody.cpp


##########################################################