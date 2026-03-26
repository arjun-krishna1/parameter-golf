"""Convert existing sp1024 shards to byte260 format.

Reads fineweb_{train,val}_*.bin files tokenized with SentencePiece (vocab 1024),
decodes each document back to text using the SP model, then re-encodes using the
PureByteTokenizer (vocab 260). Writes the result as fineweb_{train,val}_*.bin
files in the byte260 format expected by train_jepa.py.

This is a faster alternative to downloading docs_selected.jsonl and re-tokenizing
from scratch. The round-trip (SP decode -> byte encode) introduces minor whitespace
normalization artifacts but is good enough for training and development.

Usage:
    python3 data/make_byte260_from_sp1024.py

Or with custom paths:
    python3 data/make_byte260_from_sp1024.py \\
        --sp-model data/tokenizers/fineweb_1024_bpe.model \\
        --input-dir data/datasets/fineweb10B_sp1024 \\
        --output-dir data/datasets/fineweb10B_byte260 \\
        --train-shards 10
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np


DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
BOS_ID = 1  # same in both sp1024 and byte260


def load_shard(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != DATAFILE_MAGIC or int(header[1]) != 1:
        raise ValueError(f"Invalid shard header: {path}")
    num_tokens = int(header[2])
    return np.fromfile(path, dtype="<u2", count=num_tokens, offset=256 * 4)


def write_shard(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = len(tokens)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype("<u2").tobytes())


def convert_shard(
    sp,
    input_path: Path,
    output_path: Path,
    byte_offset: int = 4,
) -> tuple[int, int]:
    """Read one sp1024 shard, decode docs, re-encode as bytes, write byte260 shard."""
    tokens_sp = load_shard(input_path).tolist()

    # Split into documents by BOS boundaries (BOS_ID=1 starts each doc).
    doc_starts = [i for i, t in enumerate(tokens_sp) if t == BOS_ID]
    if not doc_starts:
        print(f"  Warning: no BOS tokens found in {input_path.name}, skipping")
        return 0, 0

    docs: list[list[int]] = []
    for idx, start in enumerate(doc_starts):
        end = doc_starts[idx + 1] if idx + 1 < len(doc_starts) else len(tokens_sp)
        # Include BOS but decode the body (BOS is token id 1, not text)
        body_ids = tokens_sp[start + 1 : end]  # skip BOS itself for SP decode
        docs.append(body_ids)

    # Decode SP token ids back to text, then byte-encode.
    out_tokens: list[int] = []
    byte_token_count = 0
    for body_ids in docs:
        if not body_ids:
            continue
        text: str = sp.decode(body_ids)
        utf8_bytes = text.encode("utf-8", errors="replace")
        byte_ids = np.frombuffer(utf8_bytes, dtype=np.uint8).astype(np.uint16) + byte_offset
        # Prepend BOS, then byte content.
        doc_tokens = np.empty(1 + len(byte_ids), dtype=np.uint16)
        doc_tokens[0] = BOS_ID
        doc_tokens[1:] = byte_ids
        out_tokens.append(doc_tokens)
        byte_token_count += len(byte_ids)

    if not out_tokens:
        return 0, 0

    combined = np.concatenate(out_tokens)
    write_shard(output_path, combined)
    return len(docs), int(combined.size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert sp1024 shards to byte260 format")
    parser.add_argument(
        "--sp-model",
        default="./data/tokenizers/fineweb_1024_bpe.model",
        help="Path to the SentencePiece .model file",
    )
    parser.add_argument(
        "--input-dir",
        default="./data/datasets/fineweb10B_sp1024",
        help="Directory containing fineweb_{train,val}_*.bin sp1024 shards",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/datasets/fineweb10B_byte260",
        help="Directory to write fineweb_{train,val}_*.bin byte260 shards",
    )
    parser.add_argument(
        "--train-shards",
        type=int,
        default=None,
        help="Max number of train shards to convert (default: all found)",
    )
    args = parser.parse_args()

    try:
        import sentencepiece as spm  # type: ignore
    except ImportError:
        raise SystemExit("sentencepiece is required: pip install sentencepiece")

    sp_model_path = Path(args.sp_model)
    if not sp_model_path.exists():
        raise SystemExit(f"SP model not found: {sp_model_path}")
    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    print(f"Loaded SP model: vocab_size={sp.vocab_size()}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert validation shards (all of them).
    val_files = sorted(input_dir.glob("fineweb_val_*.bin"))
    if not val_files:
        raise SystemExit(f"No validation shards found in {input_dir}")
    print(f"Converting {len(val_files)} validation shard(s)...")
    for src in val_files:
        dst = output_dir / src.name
        docs, toks = convert_shard(sp, src, dst)
        print(f"  {src.name} -> {dst.name}: {docs} docs, {toks:,} byte tokens")

    # Convert training shards.
    train_files = sorted(input_dir.glob("fineweb_train_*.bin"))
    if args.train_shards is not None:
        train_files = train_files[: args.train_shards]
    if not train_files:
        raise SystemExit(f"No training shards found in {input_dir}")
    print(f"Converting {len(train_files)} training shard(s)...")
    total_docs, total_toks = 0, 0
    for src in train_files:
        dst = output_dir / src.name
        docs, toks = convert_shard(sp, src, dst)
        total_docs += docs
        total_toks += toks
        print(f"  {src.name} -> {dst.name}: {docs} docs, {toks:,} byte tokens")

    print(f"\nDone. {total_docs:,} total docs, {total_toks:,} total byte tokens written to {output_dir}")
    print(f"Output shards: {len(list(output_dir.glob('fineweb_*.bin')))}")


if __name__ == "__main__":
    main()
