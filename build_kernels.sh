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

for kernel in quantize dequantize dequantize_u8 validate; do
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
    # The generated code uses: _Path(__file__).with_name("...")
    # We need: _Path(__file__).parent / "lib" / "lib<kernel>.so"
    sed -i "s|_Path(__file__).with_name(\"[^\"]*\")|_Path(__file__).parent / \"lib\" / \"${PREFIX}${kernel}${EXT}\"|" \
        "$BIND_DIR/_${kernel}_bind.py"

    echo "  -> _${kernel}_bind.py"
done

echo "Done. Libraries in $LIB_DIR, bindings in $BIND_DIR"
