CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -O2 -Iinclude -Isrc
LDFLAGS = -lm

OBJ_DIR   = build/obj
SRC_DIR   = src
BUILD_DIR = build

# Kernel objects (built by build_kernels.sh)
KERNEL_OBJS = $(OBJ_DIR)/quantize_simd.o \
              $(OBJ_DIR)/validate.o \
              $(OBJ_DIR)/dequantize_sse.o \
              $(OBJ_DIR)/dequantize_avx2.o \
              $(OBJ_DIR)/dequantize_avx512.o \
              $(OBJ_DIR)/fused_k_score.o \
              $(OBJ_DIR)/fused_v_sum.o \
              $(OBJ_DIR)/fused_attention.o \
              $(OBJ_DIR)/fused_k_score_gqa.o \
              $(OBJ_DIR)/fused_k_score_64.o \
              $(OBJ_DIR)/fused_v_sum_64.o \
              $(OBJ_DIR)/fused_k_score_gqa_64.o

# C source objects (cli.c excluded — it has main())
C_SRCS = $(filter-out $(SRC_DIR)/cli.c, $(wildcard $(SRC_DIR)/*.c))
C_OBJS = $(C_SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

LIB_STATIC = $(BUILD_DIR)/libeakv.a
LIB_SHARED = $(BUILD_DIR)/libeakv.so
CLI_BIN    = $(BUILD_DIR)/eakv

BENCH_BIN  = $(BUILD_DIR)/bench_libeakv

.PHONY: all lib cli test bench clean

all: lib cli

lib: $(LIB_STATIC) $(LIB_SHARED)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c include/eakv.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

$(LIB_STATIC): $(C_OBJS) $(KERNEL_OBJS)
	ar rcs $@ $^

$(LIB_SHARED): $(C_OBJS) $(KERNEL_OBJS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

cli: $(CLI_BIN)

$(CLI_BIN): $(SRC_DIR)/cli.c $(LIB_STATIC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(LIB_STATIC) $(LDFLAGS) -o $@

# Tests
TEST_SRCS = $(wildcard tests/test_*.c)
TEST_BINS = $(TEST_SRCS:tests/test_%.c=$(BUILD_DIR)/test_%)

test: $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "=== $$t ==="; $$t || exit 1; done

$(BUILD_DIR)/test_%: tests/test_%.c $(LIB_STATIC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Itests $< $(LIB_STATIC) $(LDFLAGS) -o $@

# Benchmarks
bench: $(BENCH_BIN)
	$(BENCH_BIN)

$(BENCH_BIN): benchmarks/bench_libeakv.c $(LIB_STATIC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(LIB_STATIC) $(LDFLAGS) -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -f $(C_OBJS) $(LIB_STATIC) $(LIB_SHARED) $(CLI_BIN) $(TEST_BINS) $(BENCH_BIN)
