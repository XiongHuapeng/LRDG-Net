# Final Ablation Protocol

The canonical ablation is:

| ID | Geometry | Degradation |
|---|---|---|
| E0_G | centered 12D | none |
| E1_A | none | absolute 3D |
| E2_R | none | local-relative 4D |
| E3_GA | centered 12D | absolute 3D |
| E4_GR | centered 12D | local-relative 4D |

Natural branch removal is used; parameter counts are reported rather than artificially forced equal.

All variants retain the same:

- source Train/Val protocol;
- source-only normalization;
- six cross-domain directions;
- seeds 42 / 2026 / 3407;
- optimizer and training budget;
- mean bag pooling.

`E4_GR` is the full LRDG-Net and must reproduce 10,898 trainable parameters.

Only the five representation-level variants reported in the manuscript are part of the canonical paper code.
