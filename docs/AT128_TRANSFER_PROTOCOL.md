# AT128 Transfer Protocol

## Research question

只回答一个问题：

> LIDAROC 上学到的 LRDG 表示，对自建 AT128 污染二分类是否具有迁移价值？

## Fixed task

- `0` = clean
- `1/2/3` = contaminated
- 不预测污染等级。

## Data split

- 独立实验单元：PCAP recording。
- 5-fold stratified cross-validation。
- 同一 PCAP 的所有连续帧始终属于同一个 fold。
- 所有方法使用完全相同的 fold 文件 `at128_transfer_folds.csv`。

## Experiment T1: Frozen LRDG + linear classifier

1. 加载 `LRDG_ALL_seed42/2026/3407`。
2. 保持 LIDAROC normalization 和 LRDG feature extractor 不变。
3. 提取每帧 64D LRDG embedding。
4. 冻结表示，仅使用 AT128 training folds 训练 Logistic Regression 二分类器。
5. 在 held-out PCAP fold 上测试。

该实验回答：LIDAROC pretrained representation 是否可迁移。

## Experiment T2: AT128 from scratch

1. 使用与 T1 完全相同的 outer PCAP folds。
2. 每个 outer training fold 内再按 PCAP 划分一个小 validation subset。
3. normalization 只在该 fold 的 AT128 inner-train 上拟合。
4. LRDG-Net 从随机初始化训练。
5. 在同一个 held-out outer fold 上测试。

该实验仅作为对照，回答 LIDAROC 预训练是否比 AT128-only training 有收益。

## Main metrics

- Macro-F1
- FPR
- FNR
- Clean Recall
- Contaminated Recall
- AUROC / AUPRC

同时保存 frame-level 和 PCAP-level 结果；论文中优先解释 PCAP-level，因为 PCAP 才是独立录制。
