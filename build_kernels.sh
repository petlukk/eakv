#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="$SCRIPT_DIR/kernels"
LIB_DIR="$SCRIPT_DIR/src/eakv/lib"
BIND_DIR="$SCRIPT_DIR/src/eakv"

mkdir -p "$LIB_DIR"

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

# Detect platform
case "$(uname -s)" in
    Linux*)  EXT=".so"; PREFIX="lib";;
    Darwin*) EXT=".dylib"; PREFIX="lib";;
    MINGW*|MSYS*|CYGWIN*) EXT=".dll"; PREFIX="";;
    *) echo "Unsupported platform" >&2; exit 1;;
esac

echo "Building eakv kernels (ea=$EA)..."

# Kernels with auto-generated Python bindings
for kernel in quantize_simd validate; do
    src="$KERNEL_DIR/${kernel}.ea"
    if [ ! -f "$src" ]; then
        echo "  WARNING: $src not found, skipping" >&2
        continue
    fi

    echo "  Compiling $kernel..."
    "$EA" "$src" --lib -o "$LIB_DIR/${PREFIX}${kernel}${EXT}"

    echo "  Generating Python bindings..."
    "$EA" bind "$src" --python

    # ea bind writes the .py next to the .ea source (or cwd); find it
    generated=""
    for candidate in "$KERNEL_DIR/${kernel}.py" "${kernel}.py"; do
        if [ -f "$candidate" ]; then
            generated="$candidate"
            break
        fi
    done

    if [ -z "$generated" ]; then
        echo "  ERROR: could not find generated ${kernel}.py" >&2
        exit 1
    fi

    # Move to package as _<kernel>_bind.py
    mv "$generated" "$BIND_DIR/_${kernel}_bind.py"

    # Fix library path: replace with_name("...") with parent / "lib" / "..."
    sed -i "s|_Path(__file__).with_name(\"[^\"]*\")|_Path(__file__).parent / \"lib\" / \"${PREFIX}${kernel}${EXT}\"|" \
        "$BIND_DIR/_${kernel}_bind.py"

    echo "  -> _${kernel}_bind.py"
done

# Multi-ISA dequantize kernels (dispatched at runtime, no bindings needed)
echo "  Compiling multi-ISA dequantize kernels..."

# SSE (f32x4) — reuse dequantize_simd.ea with SSE-only output name
"$EA" "$KERNEL_DIR/dequantize_simd.ea" --lib -o "$LIB_DIR/${PREFIX}dequantize_sse${EXT}"
echo "    -> dequantize_sse (128-bit, f32x4)"

# AVX2 (f32x8)
"$EA" "$KERNEL_DIR/dequantize_avx2.ea" --lib -o "$LIB_DIR/${PREFIX}dequantize_avx2${EXT}"
echo "    -> dequantize_avx2 (256-bit, f32x8)"

# AVX-512 (f32x16)
"$EA" "$KERNEL_DIR/dequantize_avx512.ea" --lib --avx512 -o "$LIB_DIR/${PREFIX}dequantize_avx512${EXT}"
echo "    -> dequantize_avx512 (512-bit, f32x16)"

# Fused attention kernels (AVX-512 only)
echo "  Compiling fused attention kernels..."

"$EA" "$KERNEL_DIR/fused_k_score.ea" --lib --avx512 -o "$LIB_DIR/${PREFIX}fused_k_score${EXT}"
echo "    -> fused_k_score (AVX-512, fused K dot product)"

"$EA" "$KERNEL_DIR/fused_v_sum.ea" --lib --avx512 -o "$LIB_DIR/${PREFIX}fused_v_sum${EXT}"
echo "    -> fused_v_sum (AVX-512, fused V weighted sum)"

"$EA" "$KERNEL_DIR/fused_attention.ea" --lib --avx512 -o "$LIB_DIR/${PREFIX}fused_attention${EXT}"
echo "    -> fused_attention (AVX-512, experimental fused softmax+attention)"

"$EA" "$KERNEL_DIR/fused_k_score_gqa.ea" --lib --avx512 -o "$LIB_DIR/${PREFIX}fused_k_score_gqa${EXT}"
echo "    -> fused_k_score_gqa (AVX-512, GQA loop-flipped K-score + V-sum)"

# Object files for static linking into libeakv (emit asm, then assemble)
OBJ_DIR="$SCRIPT_DIR/build/obj"
mkdir -p "$OBJ_DIR"

echo "  Compiling kernel object files for libeakv..."

emit_obj() {
    local src="$1" name="$2" flags="${3:-}"
    # shellcheck disable=SC2086
    "$EA" "$src" --emit-asm $flags
    local asm_file
    asm_file="$(basename "${src%.ea}.s")"
    gcc -c "$asm_file" -o "$OBJ_DIR/$name.o"
    rm "$asm_file"
    echo "    -> $name.o"
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

echo "Done. Libraries in $LIB_DIR, objects in $OBJ_DIR, bindings in $BIND_DIR"
