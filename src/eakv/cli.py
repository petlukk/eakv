"""eakv CLI: inspect and validate .eakv files."""

import sys
import argparse
from pathlib import Path


def cmd_inspect(args):
    from . import load
    bundle = load(args.file)

    size_mb = Path(args.file).stat().st_size / 1024 / 1024
    orig_mb = bundle.original_size / 1024 / 1024

    print(f"File:          {args.file}")
    print(f"File size:     {size_mb:.1f} MB")
    print(f"Original size: {orig_mb:.1f} MB ({bundle.orig_dtype})")
    print(f"Compression:   {bundle.compression_ratio:.1%}")
    print(f"Layers:        {bundle.n_layers}")
    print(f"Heads:         {bundle.n_heads}")
    print(f"Seq length:    {bundle.seq_len}")
    print(f"Head dim:      {bundle.head_dim}")
    print(f"Groups/layer:  {bundle.n_groups_per_layer}")
    print(f"Quant scheme:  Q4_1 (group size 64)")
    if bundle.model_hash:
        print(f"Model hash:    {bundle.model_hash}")
    if bundle.tokenizer_hash:
        print(f"Tokenizer:     {bundle.tokenizer_hash}")


def cmd_validate(args):
    from . import load, validate
    bundle = load(args.file)
    try:
        validate(bundle)
        print(f"{args.file}: OK ({bundle.n_layers} layers, {bundle.n_groups_per_layer} groups/layer)")
    except ValueError as e:
        print(f"{args.file}: FAILED - {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(prog="eakv", description="eakv KV cache tools")
    sub = parser.add_subparsers(dest="command")

    p_inspect = sub.add_parser("inspect", help="Show .eakv file metadata")
    p_inspect.add_argument("file", help="Path to .eakv file")
    p_inspect.set_defaults(func=cmd_inspect)

    p_validate = sub.add_parser("validate", help="Validate .eakv file for corruption")
    p_validate.add_argument("file", help="Path to .eakv file")
    p_validate.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
