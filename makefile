# Compiler and flags
CXX := g++
NVCC := nvcc
CXXFLAGS := -O3 -mavx2
NVCCFLAGS := -O3

# Source files
CPP_SOURCES := $(wildcard test_*.cpp)
CU_SOURCES := $(wildcard test_*.cu)

# Dependency files
DEPENDENCIES := binary_read.cpp measure_sort_time.cpp write_csv.cpp

# Generate target names by replacing 'test_' prefix with '_test' suffix
CPP_TARGETS := $(subst test_,,$(CPP_SOURCES:.cpp=_test))
CU_TARGETS := $(subst test_,,$(CU_SOURCES:.cu=_test))

# Default target
all: $(CPP_TARGETS) $(CU_TARGETS)

# Rule for building C++ executables
%_test: test_%.cpp $(DEPENDENCIES)
	$(CXX) $(CXXFLAGS) -o $@ $< $(DEPENDENCIES)

# Rule for building CUDA executables
%_test: test_%.cu $(DEPENDENCIES)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(DEPENDENCIES)

# Clean up build artifacts
clean:
	rm -f *_test

# Phony targets
.PHONY: all clean
