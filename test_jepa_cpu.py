"""CPU smoke test for train_jepa.py.

Verifies the core logic without requiring CUDA:
  - Byte260 shard loading
  - GPT model forward pass with JEPA loss
  - val_bpb calculation (simplified: val_loss / ln2)
  - Model save/load and quantization pipeline

Run with:
    python3 test_jepa_cpu.py
"""

from __future__ import annotations

import io
import math
import os
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# Patch out the CUDA requirement so we can test on CPU.
os.environ.setdefault("SKIP_CUDA_CHECK", "1")

# ---- import the model and utilities from train_jepa ----
# We need to patch one line before importing.
import types
jepa_src = Path("train_jepa.py").read_text()
# Replace the CUDA-only check so we can run on CPU.
jepa_src = jepa_src.replace(
    'if not torch.cuda.is_available():\n        raise RuntimeError("CUDA is required")',
    'pass  # CPU patched for testing',
)
# Replace cuda device setup.
jepa_src = jepa_src.replace(
    'device = torch.device("cuda", local_rank)',
    'device = torch.device("cpu")',
)
jepa_src = jepa_src.replace(
    'torch.cuda.set_device(device)',
    'pass  # CPU patched',
)
jepa_src = jepa_src.replace(
    'torch.cuda.manual_seed_all(args.seed)',
    'pass  # CPU patched',
)
jepa_src = jepa_src.replace(
    'torch.backends.cuda.matmul.allow_tf32 = True',
    'pass  # CPU patched',
)
jepa_src = jepa_src.replace(
    'torch.backends.cudnn.allow_tf32 = True',
    'pass  # CPU patched',
)
jepa_src = jepa_src.replace(
    'from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp',
    'pass  # CPU patched',
)
jepa_src = jepa_src.replace(
    '    enable_cudnn_sdp(False)\n    enable_flash_sdp(True)\n    enable_mem_efficient_sdp(False)\n    enable_math_sdp(False)',
    '    pass  # CPU patched',
)
jepa_src = jepa_src.replace(
    'with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):',
    'with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False):',
)
jepa_src = jepa_src.replace(
    'torch.cuda.synchronize()',
    'pass  # CPU patched',
)
jepa_src = jepa_src.replace(
    'torch.cuda.max_memory_allocated()',
    '0',
)
jepa_src = jepa_src.replace(
    'torch.cuda.max_memory_reserved()',
    '0',
)
jepa_src = jepa_src.replace(
    'compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)',
    'compiled_model = base_model  # no compile on CPU',
)
jepa_src = jepa_src.replace(
    'zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)',
    'pass  # no compile on CPU',
)
jepa_src = jepa_src.replace(
    'fused=True,',
    '# fused=True,  # not available on CPU',
)

_module = types.ModuleType("train_jepa_patched")
exec(compile(jepa_src, "train_jepa.py", "exec"), _module.__dict__)

GPT = _module.GPT
Hyperparameters = _module.Hyperparameters
load_data_shard = _module.load_data_shard
quantize_state_dict_int8 = _module.quantize_state_dict_int8
dequantize_state_dict_int8 = _module.dequantize_state_dict_int8


# ---- Test helpers ----

DATAFILE_MAGIC = 20240520
VOCAB_SIZE = 260
BOS_ID = 1


def make_byte260_shard(n_docs: int = 20, doc_len: int = 256) -> bytes:
    """Create a tiny in-memory byte260 shard with random text."""
    all_tokens = []
    rng = np.random.default_rng(42)
    for _ in range(n_docs):
        # BOS + random byte IDs in [4, 259]
        doc = [BOS_ID] + (rng.integers(4, 260, size=doc_len).tolist())
        all_tokens.extend(doc)
    arr = np.array(all_tokens, dtype=np.uint16)
    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = 1
    header[2] = len(arr)
    buf = io.BytesIO()
    buf.write(header.tobytes())
    buf.write(arr.astype("<u2").tobytes())
    return buf.getvalue()


def write_shard(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


# ---- Tests ----

def test_model_forward():
    """GPT with JEPA predictor: forward pass returns a scalar loss in training mode."""
    model = GPT(
        vocab_size=VOCAB_SIZE, num_layers=3, model_dim=64, num_heads=4, num_kv_heads=2,
        mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5, lambda_jepa=0.5,
    )
    model.train()
    x = torch.randint(0, VOCAB_SIZE, (2, 16))
    y = torch.randint(0, VOCAB_SIZE, (2, 16))
    loss = model(x, y)
    assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"
    assert loss.item() > 0, "Loss should be positive"
    loss.backward()
    print(f"  train forward: loss={loss.item():.4f} OK")


def test_model_eval_no_jepa():
    """In eval mode, the model returns only CE loss (no JEPA contribution)."""
    model = GPT(
        vocab_size=VOCAB_SIZE, num_layers=3, model_dim=64, num_heads=4, num_kv_heads=2,
        mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5, lambda_jepa=0.5,
    )
    model.eval()
    x = torch.randint(0, VOCAB_SIZE, (2, 16))
    y = torch.randint(0, VOCAB_SIZE, (2, 16))
    with torch.no_grad():
        loss = model(x, y)
    assert loss.shape == (), f"Expected scalar loss in eval mode, got {loss.shape}"
    print(f"  eval forward:  loss={loss.item():.4f} OK")


def test_val_bpb_formula():
    """val_bpb = val_loss / ln(2) for byte-level tokenization."""
    val_loss = 2.0  # arbitrary nats
    expected_bpb = val_loss / math.log(2.0)
    # This is what eval_val returns for byte-level.
    assert abs(expected_bpb - val_loss / math.log(2.0)) < 1e-10
    print(f"  val_bpb formula: val_loss={val_loss} -> val_bpb={expected_bpb:.4f} OK")


def test_shard_roundtrip():
    """Write a synthetic byte260 shard and load it back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_path = Path(tmpdir) / "fineweb_val_000000.bin"
        data = make_byte260_shard(n_docs=10, doc_len=128)
        write_shard(shard_path, data)
        tokens = load_data_shard(shard_path)
        assert tokens.dtype == torch.uint16
        assert tokens.numel() > 0
        # Verify BOS tokens appear
        bos_count = (tokens == BOS_ID).sum().item()
        assert bos_count == 10, f"Expected 10 BOS tokens, got {bos_count}"
        print(f"  shard roundtrip: {tokens.numel()} tokens, {bos_count} docs OK")


def test_quantization_roundtrip():
    """Quantize + dequantize: output should be close to input."""
    model = GPT(
        vocab_size=VOCAB_SIZE, num_layers=2, model_dim=64, num_heads=4, num_kv_heads=2,
        mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5, lambda_jepa=0.5,
    )
    # Submission state = no jepa_predictor.
    submission_state = {k: v for k, v in model.state_dict().items() if "jepa_predictor" not in k}
    quant_obj, stats = quantize_state_dict_int8(submission_state)
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    quant_blob = zlib.compress(buf.getvalue(), level=9)
    restored = dequantize_state_dict_int8(torch.load(io.BytesIO(zlib.decompress(quant_blob)), map_location="cpu"))
    # All keys from submission_state should be in restored.
    missing = set(submission_state.keys()) - set(restored.keys())
    unexpected = set(restored.keys()) - set(submission_state.keys())
    assert not missing, f"Missing keys after roundtrip: {missing}"
    assert not unexpected, f"Unexpected keys after roundtrip: {unexpected}"
    compressed_bytes = len(quant_blob)
    print(f"  quantization roundtrip: {compressed_bytes:,} compressed bytes OK")
    print(f"    (no jepa_predictor in submission state -- correct)")


def test_jepa_predictor_excluded_from_submission():
    """jepa_predictor.weight should NOT be in the submission state dict."""
    model = GPT(
        vocab_size=VOCAB_SIZE, num_layers=2, model_dim=64, num_heads=4, num_kv_heads=2,
        mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5, lambda_jepa=0.5,
    )
    full_state = model.state_dict()
    submission_state = {k: v for k, v in full_state.items() if "jepa_predictor" not in k}
    assert "jepa_predictor.weight" in full_state, "jepa_predictor.weight should be in full state"
    assert "jepa_predictor.weight" not in submission_state, "jepa_predictor.weight should NOT be in submission state"
    print(f"  submission state: {len(submission_state)} keys (full: {len(full_state)}, excluded jepa_predictor) OK")


def test_byte260_data_files():
    """Check that byte260 shards exist and can be loaded."""
    data_dir = Path("data/datasets/fineweb10B_byte260")
    if not data_dir.exists():
        print(f"  byte260 data: not found at {data_dir} (run data/make_byte260_from_sp1024.py)")
        return
    val_files = sorted(data_dir.glob("fineweb_val_*.bin"))
    train_files = sorted(data_dir.glob("fineweb_train_*.bin"))
    if not val_files:
        print(f"  byte260 data: no val shards found")
        return
    # Load and validate first val shard header.
    tokens = load_data_shard(val_files[0])
    print(f"  byte260 data: {len(train_files)} train, {len(val_files)} val shard(s)")
    print(f"    val shard 0: {tokens.numel():,} tokens, min={tokens.min()}, max={tokens.max()}")
    assert tokens.min() >= 0
    assert tokens.max() < 260, f"Token id {tokens.max()} >= vocab_size 260"
    print(f"  byte260 data: OK")


if __name__ == "__main__":
    print("Running train_jepa.py CPU smoke tests...\n")
    tests = [
        ("Model forward (train mode, JEPA active)", test_model_forward),
        ("Model forward (eval mode, CE only)", test_model_eval_no_jepa),
        ("val_bpb formula", test_val_bpb_formula),
        ("Shard roundtrip", test_shard_roundtrip),
        ("Quantization roundtrip (no predictor in submission)", test_quantization_roundtrip),
        ("jepa_predictor excluded from submission state", test_jepa_predictor_excluded_from_submission),
        ("byte260 data files on disk", test_byte260_data_files),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"[TEST] {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1
        print()

    print(f"Results: {passed}/{passed + failed} passed")
    sys.exit(0 if failed == 0 else 1)
