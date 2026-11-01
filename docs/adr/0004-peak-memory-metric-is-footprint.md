# Peak memory uses macOS `footprint` as the primary measure, with RSS only as a secondary measure

On macOS, compressed memory is not counted in RSS: the idle RSS of the same configuration over different launches is 689 / 4678 / 3577 MB, and under load RSS fluctuates wildly between 592 → 2536 → 170 MB, so it cannot represent usage. `footprint` (physical usage) is stable under the same conditions. Therefore peak memory = the maximum `footprint` during sampling; RSS is recorded as well, but the report notes why it is unreliable.

The `footprint` measure is a lower bound: with the default mmap, about 300 MiB of weights are file-mapped and not counted; `-lm none` raises footprint by about 294 MB (4083 vs 3789 MB). The main experiments use the default mmap, and the report gives both numbers. A single `footprint` sample takes about 0.2–0.5 s, so the sampling interval is no less than 0.5 s.

## Consequences

- This applies only to macOS. Monitoring makes "read process memory" a replaceable interface (macOS: `footprint`; Linux: `/proc/<pid>/status`), `manifest.json` and the data carry a platform label, and data from different platforms is not plotted on one figure.
