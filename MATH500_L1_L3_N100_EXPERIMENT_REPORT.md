# MATH-500 L1-L3 100-example evaluation

## What was run

The frozen seed-112 MATH-500 L1-L3 subset was enlarged from 50 to 100
examples as a strict superset. The additional 50 examples use the same
normalized level allocation as the original subset:

| Level | Original L1-L3 pool | Original ratio | n100 subset | Subset ratio |
|---|---:|---:|---:|---:|
| 1 | 43 | 18.07% | 18 | 18% |
| 2 | 90 | 37.82% | 38 | 38% |
| 3 | 105 | 44.12% | 44 | 44% |

Six Llama-3-8B-Instruct conditions were evaluated:

1. Unmodified baseline.
2. Grad on-policy expanded ranking at K=1000, K=2000, and K=4000, with
   strength 1, positive-only direction, and final-token scope.
3. SN-Tune alpha 6.
4. Raw-format IA3-SFT alpha 3, using
   `gate = 1 + 3 * (trained_gate - 1)`.

All conditions used FP32 model inference, TF32 disabled, one physical GPU per
run, batch size 16, greedy decoding, the publisher zero-shot chain-of-thought
prompt in the model's native chat template, and a 1,024 generated-token limit.
The original four runs used physical GPU 0; the added Grad K=1000 and K=2000
runs used physical GPU 1. Batch 16 was reused from the existing real-prompt
MATH-500 benchmark and is now the runner default. Scoring used deterministic
`math_verify` symbolic equivalence, not an LLM judge.

## Results

| Condition | Correct | Accuracy | Delta vs. baseline |
|---|---:|---:|---:|
| Baseline | 48/100 | 48% | -- |
| Grad on-policy K=1000 | 47/100 | 47% | -1 pp |
| Grad on-policy K=2000 | 45/100 | 45% | -3 pp |
| Grad on-policy K=4000 | 41/100 | 41% | -7 pp |
| SN-Tune alpha 6 | 46/100 | 46% | -2 pp |
| IA3-SFT alpha 3 | 45/100 | 45% | -3 pp |

Accuracy by difficulty:

| Condition | Level 1 (n=18) | Level 2 (n=38) | Level 3 (n=44) |
|---|---:|---:|---:|
| Baseline | 72.22% | 47.37% | 38.64% |
| Grad on-policy K=1000 | 66.67% | 52.63% | 34.09% |
| Grad on-policy K=2000 | 61.11% | 50.00% | 34.09% |
| Grad on-policy K=4000 | 66.67% | 47.37% | 25.00% |
| SN-Tune alpha 6 | 66.67% | 47.37% | 36.36% |
| IA3-SFT alpha 3 | 77.78% | 47.37% | 29.55% |

Paired comparison against baseline, using 10,000 seed-112 paired bootstrap
resamples and a two-sided exact McNemar test:

| Condition | Baseline errors fixed | Baseline correct broken | Paired delta 95% bootstrap CI | Exact McNemar p |
|---|---:|---:|---:|---:|
| Grad on-policy K=1000 | 9 | 10 | [-10, +7] pp | 1.0000 |
| Grad on-policy K=2000 | 14 | 17 | [-14, +8] pp | 0.7201 |
| Grad on-policy K=4000 | 12 | 19 | [-18, +4] pp | 0.2810 |
| SN-Tune alpha 6 | 13 | 15 | [-12, +9] pp | 0.8506 |
| IA3-SFT alpha 3 | 10 | 13 | [-13, +6] pp | 0.6776 |

Generation diagnostics:

| Condition | Extraction failures | Scoring errors | Repetitive outputs | Token-limit outputs | Mean generated tokens |
|---|---:|---:|---:|---:|---:|
| Baseline | 1 | 0 | 18 | 3 | 285.22 |
| Grad on-policy K=1000 | 0 | 0 | 20 | 1 | 261.56 |
| Grad on-policy K=2000 | 0 | 0 | 20 | 8 | 325.00 |
| Grad on-policy K=4000 | 1 | 0 | 19 | 7 | 314.83 |
| SN-Tune alpha 6 | 0 | 0 | 17 | 4 | 277.40 |
| IA3-SFT alpha 3 | 0 | 0 | 20 | 3 | 300.20 |

The Grad point estimate declines monotonically as K increases: 47% at K=1000,
45% at K=2000, and 41% at K=4000, versus 48% at baseline. The K=4000 loss is
concentrated at level 3. However, every paired confidence interval still
includes zero, so this n100 experiment shows a dose-ordered point estimate but
does not establish a statistically resolved safety-math trade-off by itself.

All 600 stored generations were independently rescored with direct
`math_verify` calls and reproduced the stored correctness labels exactly. The
six runs have identical example IDs and ordering. For the original four
conditions, the 50 examples inherited from the prior subset also reproduce all
200 prior responses byte-for-byte, including every correctness label.

## Artifacts

- Subset: `/workspace/xcy/dataset/math500/subsets/math500_l1_l3_seed112_n100`
- Baseline: `results/math500_l1_l3_n100_llama3_base_fp32/`
- Grad K=1000: `results/math500_l1_l3_n100_grad_onpolicy_expanded_k1000_s1_fp32/`
- Grad K=2000: `results/math500_l1_l3_n100_grad_onpolicy_expanded_k2000_s1_fp32/`
- Grad K=4000: `results/math500_l1_l3_n100_grad_onpolicy_expanded_k4000_s1_fp32/`
- SN-Tune: `results/math500_l1_l3_n100_sn_alpha6_fp32/`
- IA3-SFT alpha 3: `results/math500_l1_l3_n100_ia3_sft_snraw_alpha3_fp32/`
