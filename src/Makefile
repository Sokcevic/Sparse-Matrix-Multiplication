# ---------------------------------------
# CONFIGURATION BEGIN
# ---------------------------------------

# entry point for the program and target name
MAIN := ELL.c matrix.c Parser.c
CUDA_MAIN := cuda/ELL.cu cuda/matrix.cu cuda/Parser.cu

# assignment task file
ASSIGNMENT := main.c
ASSIGNMENT_CUDA := cuda/main.cu

# Additional flags for the compiler
CFLAGS := 

# ---------------------------------------
# CONFIGURATION END
# ---------------------------------------
NVCC := $(shell command -v nvcc)
CC := $(shell command -v gcc || command -v clang)
ifeq ($(strip $(CC)),)
    $(error Neither clang nor gcc is available. Exiting.)
endif

# all source files needed for the program
SOURCES := $(MAIN) $(ASSIGNMENT)
CUDA_SOURCES := $(CUDA_MAIN) $(ASSIGNMENT_CUDA)
TARGET := $(basename $(ASSIGNMENT))
CUDA_TARGET := $(basename $(ASSIGNMENT_CUDA))
# Default to release build for both app and library
all: debug

# Debug build
debug: CFLAGS += -g
debug: $(TARGET)

# Release build
release: CFLAGS += -O4
release: $(TARGET)

# recipe for building the program
$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $@ $(SOURCES)

# clean up
clean:
	rm -f $(TARGET) 

cuda: CFLAGS += $(CUDA_CFLAGS)
cuda: $(CUDA_TARGET)

$(CUDA_TARGET): $(CUDA_SOURCES)
ifeq ($(strip $(NVCC)),)
	$(error NVCC is not available. Exiting.)
else
	$(NVCC) $(CFLAGS) -o $@ $(CUDA_SOURCES)
endif

.PHONY: all debug release clean cuda
