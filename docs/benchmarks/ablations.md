# Runtime-control ablations

Each row changes one gMAS control while keeping the BBH samples paired. This
separates graph and plan mutation from the policy that decides when to use it.

Entry point: `python -m benchmarks.ablations.run`.

## Summary

| Control | Accuracy Δ | Token Δ | Latency Δ | Executed agents Δ | Statistical note |
| --- | ---: | ---: | ---: | ---: | --- |
| Early stopping | +1.8 pp | -52.6% | -45.0% | -3.0 | Accuracy n.s., p = .70 |
| Disabled-node handling | +3.4 pp | -44.4% | -32.1% | -2.0 | Accuracy p = .002 |
| Reachability filter | +0.5 pp | -23.1% | -34.3% | -2.0 | Accuracy n.s., p = .90 |
| Adaptive topology policy | +3.2 pp | -21.2% | -18.5% | -1.0 | Accuracy n.s., p = .251 |
| Hidden channels | -2.9 pp | +6.2% | +7.2% | 0.0 | Accuracy p = .005 |
| Multi-model routing | +9.4 pp | -4.6% | +4.2% | 0.0 | Observed accuracy effect, p = .118 |

## Variant means

Each cell is first variant / second variant, matching the comparison column.

| Control | Comparison | Accuracy | Latency (ms) | Tokens | Agents |
| --- | --- | ---: | ---: | ---: | ---: |
| Early stopping | On / Off | 74.1% / 72.3% | 19,453±8,941 / 35,386±11,203 | 2,461±1,187 / 5,196±1,305 | 2.0±1.0 / 5.0±0.0 |
| Disabled nodes | On / Off | 75.2% / 71.8% | 25,022±8,512 / 36,847±12,983 | 3,063±768 / 5,508±1,574 | 3.0±0.0 / 5.0±0.0 |
| Reachability filter | On / Off | 73.8% / 73.3% | 36,822±11,204 / 56,087±24,163 | 5,603±1,392 / 7,289±2,198 | 5.0±0.0 / 7.0±0.0 |
| Adaptive topology | Adaptive / Static | 74.8% / 71.6% | 29,681±10,752 / 36,429±12,148 | 4,298±1,662 / 5,453±1,489 | 4.0±2.0 / 5.0±0.0 |
| Hidden channels | On / Off | 69.2% / 72.1% | 31,978±21,203 / 29,836±18,124 | 4,117±2,195 / 3,875±1,867 | 5.0±0.0 / 5.0±0.0 |
| Multi-model routing | Strong+weak / Weak-only | 74.6% / 65.2% | 37,682±6,953 / 36,147±11,489 | 5,271±913 / 5,528±1,382 | 5.0±0.0 / 5.0±0.0 |

## Takeaways

- Early stop, disabled-node handling, and reachability filtering are direct cost controls with the clearest operational effect.
- The adaptive policy saves less than the direct stopping and filtering controls in this setup.
- Hidden channels are unfavorable on both cost and accuracy here.
- Multi-model routing behaves as a quality/cost trade-off, not as a speed optimization.

These results apply to the tested policies and configuration. Enabling every
adaptive feature is not a general optimization strategy.
