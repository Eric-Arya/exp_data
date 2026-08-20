# Direct activation multiplication: Llama-3 safety neurons

Updated: 2026-08-20

## Result

Direct multiplication is highly sensitive to the neuron set and multiplier. For the Llama-3
NeurIPS Instruct-vs-DPO top-20k post-MLP neurons, mild attenuation (`0.8x`) is the best tested
point: raw HarmBench ASR falls from 67.5% to 63.5%, while IFEval-32 strict prompt accuracy rises
from 68.75% to 75.0%. Amplification consistently worsens safety, reaching 76.5% ASR at `1.4x`.

| NeurIPS multiplier | Raw HarmBench ASR ↓ (n=200) | Repetitive /200 | IFEval strict prompt ↑ (n=32) |
|---:|---:|---:|---:|
| 0.6x | 64.0% | 26 | 62.50% |
| **0.8x** | **63.5%** | **19** | **75.00%** |
| 1.0x (no-op) | 67.5% | 28 | 68.75% |
| 1.2x | 70.0% | 34 | 68.75% |
| 1.4x | 76.5% | 43 | not run |

For the raw-detected SN cap-25 mask, only very mild attenuation helps. Against the matched BF16
baseline (66.0% HarmBench ASR; 69.0% full IFEval strict prompt), `0.95x` lowers ASR to 63.5% but
also lowers IFEval to 64.5%. `0.9x` is dominated by `0.95x`, and `0.8x` damages both safety and
capability.

| SN multiplier | Raw HarmBench ASR ↓ (n=200) | Repetitive /200 | IFEval strict prompt ↑ |
|---:|---:|---:|---:|
| 1.0x matched baseline | 66.0% | 29 | 69.0% (n=200) |
| **0.95x** | **63.5%** | 32 | **64.5% (n=200)** |
| 0.9x | 64.5% | 27 | 56.5% (n=200) |
| 0.8x | 79.5% | 54 | 25.0% (n=32) |

## Conclusion

Unsigned amplification is not a safety intervention for either neuron set. NeurIPS-ranked neurons
show a promising narrow `0.8x` attenuation regime, but the capability result is preliminary until
confirmed on full IFEval. SN attenuation provides clearer trade-off evidence at `0.95x`: a
2.5-point ASR improvement costs 4.5 points of IFEval accuracy, and stronger attenuation quickly
becomes destructive.

Protocol: Meta-Llama-3-8B-Instruct; all-token activation scaling; raw HarmBench-200; greedy
decoding. NeurIPS uses the Instruct-vs-DPO top-20k post-MLP ranking in FP32. SN uses the raw
ranked expanded-K/V detector mask with cap 25 per layer/structure in BF16. No fine-tuned weights
or guide model are loaded.
