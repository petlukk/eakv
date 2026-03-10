#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="$SCRIPT_DIR/kernels"
OBJ_DIR="$SCRIPT_DIR/build/obj"

mkdir -p "$OBJ_DIR"

# Find ea compiler
EA="${EA:-}"
if [ -z "$EA" ]; then
    if command -v ea &>/dev/null; then
        EA=ea
    elif [ -x "$HOME/dev/eacompute/target/release/ea" ]; then
        EA="$HOME/dev/eacompute/target/release/ea"
    elif [ -x "$HOME/dev/eacompute/target/debug/ea" ]; then
        EA="$HOME/dev/eacompute/target/debug/ea"
    else
        echo "Error: ea compiler not found. Set EA= or add to PATH." >&2
        exit 1
    fi
fi

echo "Building eakv kernel objects (ea=$EA)..."

emit_obj() {
    local src="$1" name="$2" flags="${3:-}"
    # shellcheck disable=SC2086
    "$EA" "$src" --emit-asm $flags
    local asm_file
    asm_file="$(basename "${src%.ea}.s")"
    gcc -c "$asm_file" -o "$OBJ_DIR/$name.o"
    rm "$asm_file"
    echo "  -> $name.o"
}

emit_obj "$KERNEL_DIR/quantize_simd.ea" quantize_simd
emit_obj "$KERNEL_DIR/validate.ea" validate
emit_obj "$KERNEL_DIR/dequantize_simd.ea" dequantize_sse
emit_obj "$KERNEL_DIR/dequantize_avx2.ea" dequantize_avx2
emit_obj "$KERNEL_DIR/dequantize_avx512.ea" dequantize_avx512 --avx512
emit_obj "$KERNEL_DIR/fused_k_score.ea" fused_k_score --avx512
emit_obj "$KERNEL_DIR/fused_v_sum.ea" fused_v_sum --avx512
emit_obj "$KERNEL_DIR/fused_attention.ea" fused_attention --avx512
emit_obj "$KERNEL_DIR/fused_k_score_gqa.ea" fused_k_score_gqa --avx512

echo "Done. Objects in $OBJ_DIR"
