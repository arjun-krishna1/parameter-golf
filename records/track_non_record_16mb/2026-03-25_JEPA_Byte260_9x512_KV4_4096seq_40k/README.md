# JEPA Byte-Level Submission

**Track:** Non-record, unlimited compute, 16MB artifact cap
**Final val_bpb (post int8+zlib):** 1.2093
**Baseline beaten:** 1.2244 (Naive SP-1024 Baseline)

---

## What is JEPA?

JEPA (Joint Embedding Predictive Architecture) is a self-supervised learning framework introduced by Yann LeCun (Meta). Instead of predicting raw tokens (like standard autoregressive LMs), JEPA predicts in **latent/embedding space**: a predictor network predicts the representation of future positions from the current representation, with a stop-gradient on the target. This encourages the encoder to learn richer, more predictive representations.

The challenge author specifically called out JEPA with the rule: *"no tokenizer (use byte level) to be true to JEPA"* -- byte-level operation is consistent with JEPA's philosophy of learning structure from raw signals without hand-crafted tokenization.

## Implementation

This is a "JEPA as regularizer" approach (arXiv 2509.14252, LLM-JEPA). A standard autoregressive byte-level language model is trained with an **auxiliary JEPA loss** added to the CE loss:

```
total_loss = CE_loss + lambda_jepa * JEPA_loss
```

**JEPA component:** A single `CastedLinear(512, 512)` predictor is added to the GPT class. During training, it maps hidden state `h_t` to predict `h_{t+1}`, with `h_{t+1}.detach()` as the stop-gradient target:

```python
jepa_pred = self.jepa_predictor(h[:, :-1, :])    # [B, T-1, D]
jepa_target = h[:, 1:, :].detach()               # stop-gradient
jepa_loss = F.smooth_l1_loss(jepa_pred, jepa_target)
```

The predictor is excluded from the submission artifact (not used at eval time).

## Architecture

Identical backbone to the naive SP-1024 baseline:
- 9 transformer blocks, model_dim=512
- 8 attention heads, 4 KV heads (GQA)
- ReLU² MLP, mlp_mult=2
- U-net skip connections
- RoPE, RMSNorm, logit softcap
- Tied input/output embeddings

**Key differences from baseline:**
- `VOCAB_SIZE=260` (byte-level: 4 specials + 256 raw byte values, no SentencePiece)
- `TRAIN_SEQ_LEN=4096` (4x longer sequences vs baseline's 1024 -- bytes need more context than BPE tokens)
- `val_bpb = val_loss / ln(2)` (simplified: each byte token = exactly 1 UTF-8 byte)
- +1 linear predictor layer for JEPA (262,144 params, training-only)

## Training Configuration

```bash
DATA_PATH=./data/datasets/fineweb10B_byte260   # byte-level FineWeb, 50 shards
VOCAB_SIZE=260
TRAIN_SEQ_LEN=4096
ITERATIONS=40000
MAX_WALLCLOCK_SECONDS=0                        # unlimited
WARMUP_STEPS=20
VAL_LOSS_EVERY=1000
WARMDOWN_ITERS=4000
LAMBDA_JEPA=0.1
python3 -u train_jepa.py
```

- **Hardware:** 1x NVIDIA A100 SXM 80GB
- **Total training time:** ~10.6 hours (38,331 seconds)
- **Data:** 50 FineWeb shards converted from SP-1024 to byte-level via SP decode → UTF-8 re-encode (~12.2B unique byte tokens)
- **Step speed:** ~958ms/step (single A100)

## Key Metrics

| Checkpoint | val_loss | val_bpb |
|---|---|---|
| Step 0 (random init) | 5.5762 | 8.0447 |
| Step 10000 | 0.8830 | 1.2739 |
| Step 20000 | 0.8473 | 1.2224 |
| Step 30000 | 0.8659 | 1.2494 |
| Step 36000 | 0.8484 | 1.2240 |
| Step 40000 (pre-quant) | 0.8354 | **1.2052** |
| **Final (post int8+zlib)** | **0.8382** | **1.2093** |

**Baseline:** 1.2244 BPB (Naive SP-1024, 10 min / 8×H100)

## val_bpb Metric Comparability

The challenge metric is tokenizer-agnostic bits-per-byte. For byte-level:
- Each target token = exactly 1 UTF-8 byte
- `val_bpb = val_loss_nats / ln(2) × 1.0` (tokens_per_byte = 1.0)

This is directly comparable to the SP-1024 baseline's BPB metric (which uses SP byte-length LUTs to normalize). Both measure bits needed to compress the same validation text in bytes.

## Submission Artifact

- **Total size:** 15,305,275 bytes (under 16,000,000 cap)
- **Model (int8+zlib):** 15,257,475 bytes
- **Code (`train_jepa.py`):** 47,800 bytes

## Files

- `train_jepa.py` — training script (fork of `train_gpt.py` with JEPA modifications)
- `train.log` — full training log from A100 run
- `submission.json` — leaderboard metadata
