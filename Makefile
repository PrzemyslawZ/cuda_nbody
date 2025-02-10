
## CUDA directory ##
############################################################
CUDA_INSTALL_PATH:=/usr/local/cuda
BUILD_TYPE := release

## CC COMPILER OPTIONS ##
############################################################
ALL_CCFLAGS :=
CC:=g++
CC_FLAGS:= --std=c++11
CC_LIBS:=

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
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

main.o: main.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

nbodyBench: main.o kernel.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p $(OBJ_DIR)
# $(EXEC) cp main.o kernel.o $(OBJ_DIR)
# $(RM) main.o kernel.o

run: build
	$(EXEC) ./nbodyBench

clean:
	$(RM) bin/* *.o
	$(RM)  *.o
	$(RM) ./nbodyBench


# /usr/local/cuda/bin/nvcc -ccbin g++ -I../../../Common -m64 -ftz=true --threads 0 --std=c++11 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -o bodysystemcuda.o -c bodysystemcuda.cu

# /usr/local/cuda/bin/nvcc -ccbin g++ -I../../../Common 
# -m64 -ftz=true --threads 0 --std=c++11  -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -o nbody.o -c nbody.cpp


##########################################################