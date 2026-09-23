# Final Baseline Protocol

## Comparison set

```text
Global-Statistics MLP
RangeNet-style
PointNet
PointNet++
DGCNN
PointNeXt-S-style
AutoGrAN
LRDG-Net
```

All comparison models are trained/evaluated under the same source-only session split and six-direction target protocol where applicable.

## Naming rule

- `RangeNet-style`: RangeNet/Darknet-inspired range-view encoder adapted to frame-level binary classification. It is not claimed as an exact reproduction of the original segmentation system.
- `PointNeXt-S-style`: pure-PyTorch PointNeXt-S-style frame classifier under the unified protocol. It is not claimed as an exact official OpenPoints training reproduction.
- `PointNet++`: reference SSG architecture used with the unified baseline training protocol.
- `AutoGrAN`: contamination-specific direct comparator with method-specific preprocessing/optimization retained where required.

Internal result IDs are kept stable so completed CSVs remain compatible.
