# quantization recovery and speculative decoding, from scratch

Two research pillars built on a hand-written quantization harness (round-to-nearest, GPTQ, no bitsandbytes/AutoGPTQ) for Qwen2.5-0.5B, run on Kaggle T4x2.

**Pillar 1** — post-training quantization at 4 bits fails a quality gate. Can a frozen quantized model plus a small trainable low-rank adapter (LoRA-style) recover the lost quality, and does the training objective (cross-entropy vs. distillation) matter?

**Pillar 2** — speculative decoding is provably lossless under exact acceptance. What happens to output fidelity as that criterion relaxes, and does a draft model's own quantization level change how often it gets accepted?

## Setup

- Model: `Qwen/Qwen2.5-0.5B` (target for Pillar 2: `Qwen/Qwen2.5-1.5B`)
- Hardware: Kaggle T4x2, both GPUs used concurrently for training
- Calibration/eval data: WikiText-2
- Recovery method: weights quantized once via RTN and frozen permanently; only a rank-8 LoRA adapter (~3% of total parameters) is trained
- Distillation loss compares only the teacher's top-100 tokens, not the full 151,936-token vocabulary, to keep memory bounded

## Pillar 1: does recovery work?

![recovery curve](recovery_curve.png)

Two recovery runs, same steps, same data, same starting checkpoint. **Cross-entropy diverges** — perplexity climbs from near-baseline to 180 by step 1200, ending up *worse* than doing nothing (+809% vs. the fp16 baseline). **Distillation stays stable** the whole run, landing at +10.7% — better than plain RTN (+42.9%) and GPTQ (+17.8%), though still short of the 5% gate.

![training loss curves](training_loss_curves.png)

Both losses decrease smoothly — the CE run's divergence isn't visible in its own training loss, only in held-out perplexity. Classic overfitting signature: with only 400 unique training blocks and a learning rate tuned for LoRA's zero-initialized matrices, cross-entropy training memorized the training set at the cost of general next-token prediction. Distillation's soft-target objective acted as a regularizer against exactly this.

![per-layer quantization error](per_layer_quantization_error.png)

Per-layer weight MSE, RTN vs. the CE-recovered adapter — the "recovered" bars are consistently *higher* than plain RTN, confirming the adapter moved weights in the wrong direction rather than correcting for quantization error.

![weight distribution shift](weight_distribution_shift.png)

Raw weight values barely shifted despite the catastrophic behavioral change above — a reminder that small, low-rank, compounding shifts across 24 layers can break a model's behavior without producing a visibly different weight histogram.

![quality gate outcome](quality_gate_outcome.png)

Final scoreboard against the 5% quality gate. Only int8 RTN passes. Distillation is the clear runner-up; cross-entropy is worse than not recovering at all.

**Takeaway:** recovery training is not automatically safe — the objective and learning rate matter more than the fact that *some* training is happening. Distillation's regularizing effect on a small, low-rank adapter looks like the more robust default for this setup.

## Pillar 2: is speculative decoding lossless?

![speculative decoding tradeoff](speculative_decoding_tradeoff.png)

Exact top-1 acceptance (tolerance = 1) reproduced greedy decoding byte-for-byte across every test prompt — losslessness holds, checked directly rather than assumed. Relaxing acceptance shows a **cliff, not a slope**: divergence jumps from 0 to 0.60 the instant tolerance moves from 1 to 2. For this model pair there is no safe middle ground between exactly lossless and majority-diverged.

![draft acceptance rate by precision](draft_acceptance_rate.png)

Under exact matching, the fp16 draft is accepted 68% of the time — a strong, realistic number for a same-family 0.5B/1.5B pair. The RTN-int4 draft holds up reasonably (64%); the CE-recovered draft collapses to 41%, the same overfitting damage from Pillar 1 surfacing here as a worse guesser, not just a worse standalone model.

**Caveat on the speedup numbers:** neither the baseline nor the speculative decoder in this notebook uses KV-caching — every step reprocesses the full sequence from scratch for both paths. This makes the acceptance-rate and divergence numbers trustworthy, but the measured tok/s-style "speedup" is not representative of a production implementation, where caching is exactly what turns a high acceptance rate into real wall-clock savings.

## Honest summary

- Distillation-based recovery is real and reproducible; cross-entropy recovery, at these hyperparameters, is actively harmful.
- Speculative decoding's losslessness claim is verified, not assumed — and its acceptance-rate behavior is a genuine, measurable link back to Pillar 1's quality question.
- The one clearly known limitation (no KV-cache) is scoped to the *speed* measurement only, not the correctness claims.
