#!/bin/bash
# Apply eakv Q4_1 KV cache type to llama.cpp.
#
# Usage:
#   cd /path/to/llama.cpp
#   bash /path/to/eakv/patch/apply.sh /path/to/eakv
#
# After applying:
#   cmake -B build \
#     -DCMAKE_C_FLAGS="-DGGML_USE_EAKV -I/path/to/eakv/include" \
#     -DCMAKE_CXX_FLAGS="-DGGML_USE_EAKV -I/path/to/eakv/include"
#   cmake --build build -j$(nproc)
#   ./build/bin/llama-cli -m model.gguf --cache-type-k q4_1_eakv --cache-type-v q4_1_eakv
#
set -euo pipefail

EAKV_ROOT="$(realpath "${1:?Usage: apply.sh /path/to/eakv}")"
LLAMA_ROOT="$(pwd)"

if [ ! -f "$LLAMA_ROOT/ggml/include/ggml.h" ]; then
    echo "ERROR: run from llama.cpp root directory" >&2
    exit 1
fi
if [ ! -f "$EAKV_ROOT/include/eakv_ggml.h" ]; then
    echo "ERROR: eakv not found at $EAKV_ROOT" >&2
    exit 1
fi

echo "Patching llama.cpp at $LLAMA_ROOT"
echo "  eakv source: $EAKV_ROOT"
echo ""

# ---------- 1. ggml.h — add enum entry ----------
echo "  [1/6] ggml.h — GGML_TYPE_Q4_1_EAKV = 40"

sed -i '/GGML_TYPE_MXFP4.*=.*39/a\        GGML_TYPE_Q4_1_EAKV = 40, // eakv Q4_1 (group_size=64, AVX-512)' \
    "$LLAMA_ROOT/ggml/include/ggml.h"

sed -i 's/GGML_TYPE_COUNT   = 40/GGML_TYPE_COUNT   = 41/' \
    "$LLAMA_ROOT/ggml/include/ggml.h"

# ---------- 2. ggml.c — add type_traits entry ----------
echo "  [2/6] ggml.c — type_traits (to_float, from_float_ref)"

sed -i '/#include "ggml-impl.h"/a\
#ifdef GGML_USE_EAKV\
#include "eakv_ggml.h"\
#endif' "$LLAMA_ROOT/ggml/src/ggml.c"

sed -i '/\[GGML_TYPE_MXFP4\]/,/},/{
    /},/a\#ifdef GGML_USE_EAKV\
    [GGML_TYPE_Q4_1_EAKV] = {\
        .type_name                = "q4_1_eakv",\
        .blck_size                = 64,\
        .type_size                = 40,\
        .is_quantized             = true,\
        .to_float                 = (ggml_to_float_t) eakv_dequantize_row,\
        .from_float_ref           = (ggml_from_float_t) eakv_quantize_row,\
    },\
#endif
}' "$LLAMA_ROOT/ggml/src/ggml.c"

# ---------- 3. ggml-cpu.c — add type_traits_cpu entry ----------
echo "  [3/6] ggml-cpu.c — type_traits_cpu (vec_dot, from_float)"

sed -i '/#include "ggml-cpu-impl.h"/a\
#ifdef GGML_USE_EAKV\
#include "eakv_ggml.h"\
#endif' "$LLAMA_ROOT/ggml/src/ggml-cpu/ggml-cpu.c"

sed -i '/\[GGML_TYPE_MXFP4\]/,/},/{
    /},/a\#ifdef GGML_USE_EAKV\
    [GGML_TYPE_Q4_1_EAKV] = {\
        .from_float               = (ggml_from_float_t) eakv_quantize_row,\
        .vec_dot                  = (ggml_vec_dot_t) eakv_vec_dot_q4_f32,\
        .vec_dot_type             = GGML_TYPE_F32,\
        .nrows                    = 1,\
    },\
#endif
}' "$LLAMA_ROOT/ggml/src/ggml-cpu/ggml-cpu.c"

# ---------- 4. arg.cpp — add to allowed cache types ----------
echo "  [4/6] arg.cpp — kv_cache_types"

sed -i '/GGML_TYPE_Q5_1,/a\#ifdef GGML_USE_EAKV\
    GGML_TYPE_Q4_1_EAKV,\
#endif' "$LLAMA_ROOT/common/arg.cpp"

# ---------- 5. Copy eakv source into ggml-base ----------
echo "  [5/6] Copying eakv_ggml.h and ggml_type.c into ggml tree"

cp "$EAKV_ROOT/include/eakv_ggml.h" "$LLAMA_ROOT/ggml/src/eakv_ggml.h"
cp "$EAKV_ROOT/include/eakv_ggml.h" "$LLAMA_ROOT/ggml/src/ggml-cpu/eakv_ggml.h"
cp "$EAKV_ROOT/src/ggml_type.c"     "$LLAMA_ROOT/ggml/src/eakv-ggml-type.c"

# ---------- 6. CMakeLists.txt — add source to ggml-base ----------
echo "  [6/6] CMakeLists.txt — adding eakv-ggml-type.c to ggml-base"

sed -i '/ggml-quants.h/a\            eakv-ggml-type.c' \
    "$LLAMA_ROOT/ggml/src/CMakeLists.txt"

echo ""
echo "Done. Build with:"
echo ""
echo "  cmake -B build \\"
echo "    -DCMAKE_C_FLAGS=\"-DGGML_USE_EAKV -I$EAKV_ROOT/include\" \\"
echo "    -DCMAKE_CXX_FLAGS=\"-DGGML_USE_EAKV -I$EAKV_ROOT/include\""
echo "  cmake --build build -j\$(nproc)"
echo ""
echo "Run with:"
echo "  ./build/bin/llama-cli -m model.gguf --cache-type-k q4_1_eakv --cache-type-v q4_1_eakv"
