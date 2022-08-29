#-- Set default include makefile
MACHINE ?= $(shell builds/machine.sh)
TYPE    ?= default

include builds/make.host.$(MACHINE)
# include builds/make.type.$(TYPE)

DIRS     := src 

SUFFIX ?= .$(TYPE).$(MACHINE)

CFILES   := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.c))
CPPFILES := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.cpp))
GPUFILES := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.cu))

# Build a list of all potential object files so cleaning works properly
CLEAN_OBJS := $(subst .c,.o,$(CFILES)) \
              $(subst .cpp,.o,$(CPPFILES)) \
              $(subst .cu,.o,$(GPUFILES))

OBJS     := $(subst .c,.o,$(CFILES)) \
            $(subst .cpp,.o,$(CPPFILES)) \
            $(subst .cu,.o,$(GPUFILES))

#-- Set default compilers and flags
CC                ?= cc
CXX               ?= CC

CFLAGS_OPTIMIZE   ?= -Ofast
CXXFLAGS_OPTIMIZE ?= -Ofast -std=c++11
GPUFLAGS_OPTIMIZE ?= -g -O3 -std=c++11
BUILD             ?= OPTIMIZE

CFLAGS            += $(CFLAGS_$(BUILD))
CXXFLAGS          += $(CXXFLAGS_$(BUILD))
GPUFLAGS          += $(GPUFLAGS_$(BUILD))

#-- Add flags and libraries as needed
CFLAGS   += $(DFLAGS) -Isrc
CXXFLAGS += $(DFLAGS) -Isrc
GPUFLAGS += $(DFLAGS) -Isrc


CXXFLAGS += -I$(ROCM_PATH)/include
GPUFLAGS += -I$(ROCM_PATH)/include
LIBS += -L$(ROCM_PATH)/lib -lhiprand


DFLAGS    += -DO_HIP
CXXFLAGS  += $(HIPCONFIG)
GPUCXX    ?= hipcc
GPUFLAGS  += -std=c++11 -Wall -ferror-limit=1 -fPIE
LD        := $(CXX)
LDFLAGS   := $(CXXFLAGS)
LIBS      += -L$(ROCM_PATH)/lib -lamdhip64 -lhsa-runtime64


.SUFFIXES: .c .cpp .cu .o

EXEC := bin/ising3D$(SUFFIX)

# # Get the git hash and setup macro to store a string of all the other macros so
# # that they can be written to the save files
# DFLAGS      += -DGIT_HASH='"$(shell git rev-parse --verify HEAD)"'
# MACRO_FLAGS := -DMACRO_FLAGS='"$(DFLAGS)"'
# DFLAGS      += $(MACRO_FLAGS)

$(EXEC): prereq-build $(OBJS)
	mkdir -p bin/ && $(LD) $(LDFLAGS) $(OBJS) -o $(EXEC) $(LIBS)
	eval $(EXTRA_COMMANDS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(GPUCXX) $(GPUFLAGS) -c $< -o $@ 

.PHONY: clean

clean:
	rm -f $(CLEAN_OBJS)
	-find bin/ -type f -executable -name "ising3D.*.$(MACHINE)*" -exec rm -f '{}' \;

clobber: clean
	find . -type f -executable -name "ising3D*" -exec rm -f '{}' \;
	-find bin/ -type d -name "t*" -prune -exec rm -rf '{}' \;
	rm -rf bin/ising3D.*tests*.xml

prereq-build:
	builds/prereq.sh build $(MACHINE)
prereq-run:
	builds/prereq.sh run $(MACHINE)
