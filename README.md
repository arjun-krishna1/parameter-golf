<img width="3840" height="1280" alt="1920x640-discord" src="https://github.com/user-attachments/assets/90607b26-171f-476a-90ae-69b9dbb7cb30" />

<br>
<br>

**OpenAI Model Craft Challenge: Parameter Golf** is a challenge to train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s, evaluated by compression on the FineWeb validation set (tokenizer-agnostic, bits per byte).

This challenge is heavily inspired by the [NanoGPT Speedrunning](https://github.com/KellerJordan/modded-nanogpt) challenge, where participants compete to train a model that reaches 3.28 FineWeb validation loss as quickly as possible. We're excited to see how optimizing for a parameter-constrained setting pushes people toward unique architectures (test-time compute, aggressive parameter tying, depth recurrence, low-rank training, ...), compression schemes (low precision, QAT, bitnets, novel tokenizers, ...), and other creative submissions (test-time training, long context, megakernels ...). 

If you're familiar with [neural scaling laws](https://arxiv.org/abs/2001.08361), you can consider this challenge a form of L(N) optimization, where the objective is to optimize the lowest loss given a fixed number of parameters (N) unconstrained by data, compute, steps, or architecture. Challenges like the [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt), which optimizes for a form of L(T) (~lowest time given constrained loss) or the [NanoGPT Slowrun](https://github.com/qlabs-eng/slowrun), which optimizes for L(D) (lowest loss given constrained dataset size), can be thought of as equivalent challenges in this family.

Ideally, we'd allow for submissions to use arbitrary computational resources. But in order to make the challenge not inaccessibly expensive, we're limiting *leaderboard submissions* to 10 minutes on 8xH100s. However, we'd still love to see submissions that don't meet the compute limitation requirements in our 'Non-record Submissions' section: We're excited to see people push the infinite frontier of parameter limited performance as well.

We also know compute is expensive, so **OpenAI is sponsoring $1,000,000 in compute credits** to help people get started training their models. To request a compute grant, use this form: [Request a Compute Grant](https://openai.com/index/parameter-golf/#credit-form).

## Participant Form

If you enjoy solving very difficult technical problems, please introduce yourself via the [Challenge Participant Form](https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf). It helps us attribute challenge submissions and reach out about opportunities with OpenAI. _Completing the form is not required to participate._

Many researchers at OpenAI first distinguished themselves through elite mathematics and programming competitions. The Model Craft Challenge is designed in that spirit: testing the ability to tackle unfamiliar problems with creativity and rigor, qualities we believe are essential for frontier AI research.

In June, we plan to hire a small cohort of early-career researchers, targeting current undergraduate students and recent graduates, including Olympiad medalists and elite competitors. For exceptional participants, the challenge may also serve as a way to stand out to OpenAI researchers and recruiters.

The challenge runs from March 18th to April 30th. 

Happy training!

## Leaderboard


| Rank | Run | Score | Author | Summary | Date | Info |
|-----:|-----|------:|--------|---------|------|------|
| 1 | Naive Baseline | 1.2244 | Baseline | 9layer 512dim 1024vocab TiedEmbeddings 4 KV heads | 2026-03-18 | [info](records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md) |

#### Notable Non-Record Runs

| Run | Score | Author | Summary | Date | Info |
|-----|------:|--------|---------|------|------|
| 4-Hour Baseline | 1.2074 | Will DePue | Testing unlimited compute, 4 hours on 8xH100 | 2026-03-18 | [info](records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md) |

## Getting Started

### Training Your First Model (Mac with Apple Silicon)

If you have an Apple laptop or desktop with Apple Silicon, we've set up a simple MLX training script to help you start iterating locally.

If you don't have a Mac with Apple Silicon, you can run an adapted version of this script without MLX support. Just ask [Codex](https://openai.com/codex/) to refactor it; the change is straightforward. It may still be fairly slow, so we recommend jumping straight to cloud GPUs with Runpod.

First, clone the repository, create a fresh Python environment, and install the packages needed for the MLX path plus dataset download:

```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
```

Download our cached version of FineWeb with the 1024-token vocabulary:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

This populates `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/`.
By default this downloads the full validation split plus 80 training shards (8B tokens). For a smaller local smoke subset, pass `--train-shards 1`, for example `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`.

Then run a small MLX training job:

```bash
RUN_ID=mlx_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```

Validation always runs on the full `fineweb_val_*` split, which is the fixed first-50k-document set. The smoke command above skips periodic validation and just prints the final `val_loss` and `val_bpb` once at the end.

### Scaling Up to a Remote Machine

Once you're happy with your local tests, or you want more compute, switch to a remote CUDA machine.

You can rent GPUs from anywhere, but OpenAI is partnering with Runpod to make setup as easy as possible.  

#### Launching a 1xH100 Pod

1. First, [create a Runpod account](https://console.runpod.io/deploy). You should also set up an SSH key in the Settings tab on the left so you can connect to your remote machine. If you're new to this, ask Codex to help you set it up.

2. Once you've set up your account, create a new GPU Cloud Pod. You can choose whichever GPU SKU you'd like. Final leaderboard submissions must run in under 10 minutes on 8xH100s (specifically the SXM variant), but we strongly recommend testing and running experiments on cheaper SKUs first, since an 8xH100 box can cost around $20/hour.

3. Let's start with a 1xH100 pod. Deploy using the official Parameter Golf template: [Launch Template](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th). Enable SSH terminal access, leaving the other settings at their defaults. Deploy your pod and SSH into it once it's up. You should land in `/workspace/`.

On your remote machine, clone the repo onto local disk. All Python dependencies are already pre-installed in the image.

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
```

Download our cached version of FineWeb. We'll use the 1024-token vocabulary for now.

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

This defaults to the full validation split plus 80 training shards (8B tokens). If you only want a smaller subset while iterating, pass `--train-shards N`, for example `--train-shards 1`.

Launch your first training run. Note that we're passing `nproc_per_node=1` because we're running on a single H100 GPU in this case.

```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

By default, `train_gpt.py` keeps its ~10 minute wallclock cap. If you want a longer run, override it explicitly, for example `MAX_WALLCLOCK_SECONDS=0`.

By default, this command prints `train_loss` step logs during training and prints `val_loss`, `val_bpb`, and compressed model size in the final `final_int8_zlib_roundtrip` lines at the end. If you want periodic validation logs during the run, set `VAL_LOSS_EVERY`, for example `VAL_LOSS_EVERY=200`. For the baseline config, the final `val_bpb` should land around ~1.2 with a compressed model size under 16MB.

For dataset export, tokenizer export, and docs-cache rebuild instructions, see [data/README.md](data/README.md).


## FAQ

**What exactly counts toward the 16MB artifact size?**

The submission artifact is computed as code bytes plus compressed model bytes. All counted code should live in the `train_gpt.py` script.
The cap is decimal 16MB, i.e. 16,000,000 total bytes, not 16 MiB / 16,777,216 bytes.
No external downloads, training dataset access, or network calls are allowed during evaluation. The artifact must be fully self-contained and reproducible.

**Are scores independently verified by OpenAI?**

We're not automatically verifying every submission, but we will verify the top leaderboard entries over time. Any non-reproducible results can be disqualified, and issues reproducing submissions should be raised on the PR.

**What counts as 'external compute'? For example, is it fair to tune my hyperparameters offline?**

There's no perfectly clear answer here and it's hard to draw a clean line around what does or does not count as external compute. For now, we're reserving the right to disqualify runs that are not in the spirit of the challenge. Tuning your Adam hyperparameters across a bunch of runs is fine, but if there's evidence that you're sneaking in additional compute unfairly, such as brute-forcing ridiculous seeds, we won't allow it. Use your best judgment and there's no penalty for asking questions.

**What are the restrictions on evaluation?**

We won't accept submissions that take more than 10 minutes on 8xH100 to evaluate (Note: This limit is in addition to the 10 minutes of training time allowed!), but otherwise you're free to evaluate however. As with modded-nanogpt, we allow evaluation at any sequence length. And, obviously, you aren't allowed to access any training data during evaluation, unless you pay for those bits in the <16MB limit. We encourage competitors to push the bounds of evaluation methods as aggressively as with training methods.

## Submission Process

New SOTA records must fulfill the following criteria:

1. They must beat the existing SOTA by at least 0.005 nats. As in modded-nanogpt, because of inter-run variance all submissions must provide enough run logs to show at `p < 0.01` that they achieved the required 0.005-nat improvement. For submissions that improve speed through systems optimization without changing the ML, this requirement is waived.

2. If changes are made to the tokenizer or dataset, prove with certainty that the val_bpb is correctly calculated. Submissions that edit the tokenizer will be examined much more carefully, since bugs may unjustly improve your score.

3. Reproducibly run in under 10 minutes on 8xH100s.

All submissions should be made as a pull request that only adds a new folder to the appropriate `/records` subfolder and includes the following files. Submissions without the full set of requirements will not be accepted.

1. A README.md file that explains the submission in reasonable detail.

2. A `submission.json` file (see the example runs) that includes your name, GitHub ID, `val_bpb`, and related metadata.

3. A train log, automatically produced by your script.

4. A `train_gpt.py` script and any other dependencies. Note: this must successfully compile and run within the records folder. Broken scripts will not be accepted.

### Non-record Submissions

Submissions are also open to unique and interesting approaches that might not beat the existing SOTA, but still satisfy the 16MB artifact limit. We strongly encourage participants to submit implementations for weird or out-of-the-box ideas, in-progress or unoptimized solutions, so long as they run successfully, or even interesting negative results. We're excited to see what you come up with. We'll still maintain a high bar for non-record submissions, so be sure to justify your ideas and results in detail when submitting.

We also accept non-record submissions to an unlimited compute track for runs that are not intended to meet the 10-minute cutoff. Just note as such in your README file.

Non-record submissions should be made in the same fashion as SOTA records, as described above.

#### PRs on Core Code

The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but the best models should stay in the `/records` folder.

## Support


Join the [OpenAI Discord server](https://discord.com/invite/openai) and visit the Parameter Golf channels (#parameter-golf-discussions, #parameter-golf-announcements) and ask questions.

This repository adapts code from `modded-nanogpt`, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.

---

## Experiment Log

### Experiment 0: Baseline Run (8xH100, sp1024)

- Config: 9-layer, 512-dim, 1024 vocab, tied embeddings, GQA (8 heads / 4 KV), seed 1337. 524,288 batch tokens, 1024 seq len, 20k iterations max, 600s wallclock cap.
- Params: 17,059,912
- Result: Hit wallclock cap at step 11,701/20,000 (~600s). Final val_bpb 1.2228 (pre-roundtrip), **1.2298 after int8+zlib roundtrip**. val_loss 2.0764.
- Artifact size: 15,809,501 B compressed (int8+zlib), 15,857,187 B total submission (incl. code). Comfortably under the 16MB cap with ~143 KB to spare.
- Notes: Loss was still decreasing when wallclock hit — more steps would likely help. Step avg drifted from ~44ms to ~51ms over the run. The int8+zlib roundtrip costs ~0.007 bpb (1.2228 → 1.2298).

### Experiment 1: zstd vs zlib Compression for Model Artifact

- Hypothesis: The baseline pipeline compresses the int8 model checkpoint with zlib. zstd (Zstandard) is a newer compression algorithm that often achieves better ratios on binary data. Switching to zstd could reclaim bytes within the 16MB artifact cap, effectively allowing more parameters for the same budget.

- Setup: Decompressed the baseline `final_model.int8.ptz` (zlib-compressed) back to raw bytes, then recompressed with zstd at levels 1–22. Compared output sizes against the original zlib blob. See `compression_comparison.py`.

- Results:

| Format | Level | Compressed Size | Savings vs zlib |
|--------|------:|----------------:|----------------:|
| zlib (baseline) | default | 15,809,501 B | — |
| zstd | 1 | 15,787,382 B | 22,119 B |
| zstd | 3 | **15,777,718 B** | **31,783 B** |
| zstd | 6 | 15,780,078 B | 29,423 B |
| zstd | 10 | 15,779,708 B | 29,793 B |
| zstd | 15 | 15,779,077 B | 30,424 B |
| zstd | 19 | 15,780,855 B | 28,646 B |
| zstd | 22 | 15,780,870 B | 28,631 B |

Raw uncompressed size: 17,224,025 bytes (17.22 MB).

Best result was **zstd level 3**, saving ~31.8 KB over zlib — enough room for roughly 31,783 extra int8 parameters.

- Recommendation: The savings are real but modest (~0.2% of the artifact size). Worth adopting if we're pushing right up against the 16MB cap and need every byte, since the code change is trivial (swap `zlib.compress` for `zstd.ZstdCompressor(level=3).compress`). However, if we're comfortably under budget, the complexity of adding a `zstandard` dependency isn't justified — the ~32K of freed space is unlikely to meaningfully move val_bpb on its own.

### Experiment 2: Recurrence & Wider Model (3×5 loops, 704d)

- Hypothesis: Depth recurrence (looping a small set of unique transformer blocks multiple times) lets you trade unique parameters for effective depth. Using fewer unique layers at a wider hidden dimension could improve per-step learning quality enough to offset the higher per-step cost.

- Setup: 3 unique core layers looped 5 times (15 effective layers), 704-dim, same 1024 vocab / tied embeddings / GQA setup. 8xH100, 600s wallclock cap.

- Results:

| | Baseline | Recurrence (3×5, 704d) |
|---|---|---|
| **final val_bpb (int8)** | **1.2298** | **1.2874** |
| Steps completed | 11,701 | 4,768 |
| step_avg | ~51ms | ~126ms |
| int8+zlib size | 15.8 MB | 10.3 MB |

- Outcome: **The recurrence model lost by 0.058 bpb.** It completed only 41% of training steps (4,768 vs 11,701) because each step is ~2.5× more expensive. The wider model learns better per-step, but not nearly enough to compensate for 59% fewer steps.

- Positive: Compressed size is only 10.3 MB — **5.7 MB of headroom** under the 16 MB limit. The architecture works correctly. This is purely a speed-vs-quality tradeoff problem.

- Next steps: The core issue is that 15 effective layers at 704d costs too many FLOPs. Need to reduce per-step cost while keeping the width advantage.

  1. **3×3 at 704d** — 9 effective layers (same as baseline), 704d wide, 3 unique blocks. FLOPs ratio vs baseline: `9 × 704² / (9 × 512²) ≈ 1.89×`. Estimated ~94ms/step, ~6,400 steps. Still 45% fewer steps than baseline, but per-step learning at 704d might compensate.
  2. **3×3 at 576d** — 9 effective layers, 576d, 3 unique blocks. FLOPs ratio: `9 × 576² / (9 × 512²) ≈ 1.27×`. Estimated ~63ms/step, ~9,500 steps. Only 19% fewer steps than baseline; modest width gain but nearly the same training budget. ~7.6M params, compresses to ~7 MB.
  3. **4×2 at 704d** — 8 effective layers, 704d, 4 unique blocks. More unique blocks gives more representational diversity (the main weakness of 3 shared blocks). FLOPs ratio: `8 × 704² / (9 × 512²) ≈ 1.68×`. Estimated ~84ms/step, ~7,100 steps. ~14.6M params, compresses to ~13 MB.

  Recommended order: Option 1 first (most direct test of width-for-speed), then Option 2 (conservative fallback with near-baseline step count) if it loses.

### Experiment 3: Recurrence with Reduced Depth (3×3 loops, 704d)

- Hypothesis: The 3×5 config lost because 15 effective layers at 704d was too expensive per step (~126ms), limiting total training steps. Reducing to 3×3 (9 effective layers — same depth as baseline) at 704d should cut per-step cost enough to get significantly more steps, while keeping the width advantage.

- Setup: 3 unique core layers looped 3 times (9 effective layers), 704-dim, same 1024 vocab / tied embeddings / GQA setup. 8xH100, 600s wallclock cap.

- Results:

| | Baseline | 3×5 (704d) | 3×3 (704d) |
|---|---|---|---|
| **final val_bpb (int8)** | **1.2298** | **1.2874** | **1.2938** |
| Steps completed | 11,701 | 4,768 | 7,286 |
| step_avg | ~51ms | ~126ms | ~82ms |
| int8+zlib size | 15.8 MB | 10.3 MB | 10.3 MB |
| model_params | 17.1M | 11.2M | 11.1M |

- Val bpb trajectory (vs baseline at same step count):

| Step | Baseline | 3×3 (704d) | Gap |
|------|----------|------------|-----|
| 1000 | 1.3839 | 1.4334 | +0.050 |
| 2000 | 1.3244 | 1.3730 | +0.049 |
| 3000 | 1.3001 | 1.3482 | +0.048 |
| 4000 | 1.2849 | 1.3336 | +0.049 |
| 5000 | 1.2748 | 1.3239 | +0.049 |
| 6000 | 1.2689 | 1.3187 | +0.050 |
| 7000 | 1.2625 | 1.2921 | +0.030 |

- Outcome: **The 3×3 config lost by 0.064 bpb — worse than the 3×5.** Despite getting 53% more steps than 3×5 (7,286 vs 4,768), the shallower depth (9 vs 15 effective layers) lost more quality than the extra steps recovered. Step time (~82ms) was 1.6× baseline, not the estimated 1.89×, confirming FLOPs-to-walltime scaling is sublinear.

- Key insight: **3 unique blocks is the bottleneck, not depth or width.** The 3×3 was consistently ~0.05 bpb worse than baseline at every step count. This gap was nearly constant from step 1000 to 6000, meaning the model's per-step learning efficiency is fundamentally capped by having only 3 distinct transformer blocks. The wider dimension (704 vs 512) does not compensate for the lost representational diversity of going from 9 unique blocks to 3.

- Implication: Depth recurrence with very few unique blocks (3) is a dead end for beating this baseline. The parameter savings (5.7 MB headroom) are real but unusable — the model can't learn as efficiently per step regardless of how many extra steps it gets.

### Next Steps — Strategy Reassessment

The recurrence experiments (3×5 and 3×3 at 704d) established clear findings:
1. **Width at the expense of unique blocks does not pay off.** 704d × 3 unique blocks is consistently ~0.05 bpb/step worse than 512d × 9 unique blocks.
2. **Depth via recurrence helps per-step quality** (3×5 beat 3×3 in final bpb despite fewer steps), but the compute cost eliminates the advantage.
3. **Step time dominance**: Within a fixed wallclock budget, total steps completed is the primary driver of final quality. Any config that significantly slows step time loses.

Promising directions to explore:

1. **Wider baseline without recurrence** — Keep 9 unique layers, increase model_dim from 512 to 544–576. No recurrence overhead, so step time stays closer to baseline (~57–65ms). Tests whether width helps when unique block count is held constant at 9. ~12–13M params, compresses to ~12 MB (comfortably under 16 MB). This isolates the width variable.

2. **Recurrence with more unique blocks** — e.g. 5×2 at 512d (10 effective layers, 5 unique blocks, ~57ms/step) or 6×2 at 512d (12 effective, 6 unique, ~68ms/step). More unique blocks should close the per-step quality gap while still benefiting from mild recurrence.

3. **Orthogonal improvements** — Learning rate schedule tuning, sequence length changes at eval time, tokenizer experiments, or other strategies mentioned in the challenge description (test-time compute, QAT, etc.) that don't require the recurrence architecture to work.

Recommended order: Option 1 first (simplest test, isolates width vs block count), then Option 2 if width alone helps.
