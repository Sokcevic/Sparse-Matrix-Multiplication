# ---------------------------------------
# CONFIGURATION BEGIN
# ---------------------------------------

# entry point for the program and target name
MAIN := ELL.c matrix.c Parser.c multiply.c
CUDA_MAIN := cuda/multiply.cu

# assignment task file
ASSIGNMENT := main.c

# test program
TEST := test.c

# Additional flags for the compiler
CFLAGS := -std=c17 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -fopenmp
# Restrict the CPU build to SSE (up to SSE4.2), no x87, MMX or AVX. x86-64 uses SSE
# for floating point by default, these flags turn any violation into an error.
ISAFLAGS := -mfpmath=sse -mno-80387 -mno-mmx -mno-avx
LDLIBS := -lm

# Flags for nvcc (only used by make cuda / make test-cuda)
NVCCFLAGS := -O3 -I. -DWITH_CUDA
CUDA_LDLIBS := -lgomp -lm

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
HEADERS := $(wildcard *.h)
TARGET := $(basename $(ASSIGNMENT))
TEST_TARGET := ell_test
CUDA_TARGET := $(TARGET)_cuda
CUDA_TEST_TARGET := $(TEST_TARGET)_cuda

# object files of the CUDA build: the C sources compiled with -DWITH_CUDA
CUDA_BUILD := build-cuda
CUDA_OBJECTS := $(CUDA_BUILD)/multiply_cu.o
CUDA_C_OBJECTS := $(patsubst %.c,$(CUDA_BUILD)/%.o,$(MAIN) $(ASSIGNMENT) $(TEST))

CFLAGS += $(ISAFLAGS)

# Default to release build for both app and library
all: release

# Debug build
debug: CFLAGS += -g -O0
debug: $(TARGET)

# Release build
release: CFLAGS += -O3
release: $(TARGET)

# build and run the tests
test: CFLAGS += -g -O2
test: $(TEST_TARGET)
	./$(TEST_TARGET)

# CUDA build: the program with the additional GPU versions -V 3 and -V 4
cuda: check-nvcc $(CUDA_TARGET)

# build and run the tests including the GPU versions
test-cuda: check-nvcc $(CUDA_TEST_TARGET)
	./$(CUDA_TEST_TARGET)

check-nvcc:
	@test -n "$(NVCC)" || (echo "NVCC not installed. Exiting."; exit 1)

# clean up
clean:
	rm -rf $(TARGET) $(TEST_TARGET) $(CUDA_TARGET) $(CUDA_TEST_TARGET) \
		$(CUDA_BUILD)

# recipe for building the program
$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $(SOURCES) $(LDLIBS)

$(TEST_TARGET): $(MAIN) $(TEST) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $(MAIN) $(TEST) $(LDLIBS)

$(CUDA_BUILD)/%.o: %.c $(HEADERS)
	@mkdir -p $(CUDA_BUILD)
	$(CC) $(CFLAGS) -O3 -DWITH_CUDA -c -o $@ $<

$(CUDA_BUILD)/multiply_cu.o: $(CUDA_MAIN) $(HEADERS)
	@mkdir -p $(CUDA_BUILD)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(CUDA_TARGET): $(filter-out $(CUDA_BUILD)/test.o,$(CUDA_C_OBJECTS)) \
		$(CUDA_OBJECTS)
	$(NVCC) -o $@ $^ $(CUDA_LDLIBS)

$(CUDA_TEST_TARGET): $(filter-out $(CUDA_BUILD)/main.o,$(CUDA_C_OBJECTS)) \
		$(CUDA_OBJECTS)
	$(NVCC) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: all debug release clean cuda test test-cuda check-nvcc
