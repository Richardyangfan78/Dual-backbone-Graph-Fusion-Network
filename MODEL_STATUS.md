# Model Status

Last updated: 2026-07-06

This document records the current reporting model directories for the inorganic
chalcohalide multitask experiments. The dual-backbone checkpoints listed below
are the latest models to use for reporting and downstream work.

## Latest Official Models

| Model | Latest checkpoint directory | BG MAE | GT F1 | EH F1 |
|---|---|---:|---:|---:|
| MACE+ALIGNN | `checkpoints/dual_backbone_inorg_preAlignSplit_20260702_080248` | 0.2218 | 0.8476 | 0.9533 |
| MACE+M3GNet | `checkpoints/mace_m3gnet_inorg_oldm3gnet_20260705_081640` | 0.2587 | 0.8537 | 0.9576 |

## Backbone Checkpoints Used

- MACE+ALIGNN uses the archived best-BG fusion checkpoint from
  `checkpoints/dual_backbone_inorg_preAlignSplit_20260702_080248`. This
  checkpoint contains the full fused model state and is the latest reporting
  checkpoint for MACE+ALIGNN.
- MACE+M3GNet loads the restored M3GNet backbone from
  `checkpoints/old_split_before_mace_alignn_20260630_082959/m3gnet_mt_inorg`.
- Both latest dual models use the same aligned split directory:
  `Data/Inorganic_datasets/splits_mace_alignn`.

Fold sizes for the aligned split:

| Fold | Train | Val | Test |
|---|---:|---:|---:|
| 0 | 1202 | 212 | 354 |
| 1 | 1202 | 212 | 354 |
| 2 | 1202 | 212 | 354 |
| 3 | 1203 | 212 | 353 |
| 4 | 1203 | 212 | 353 |

## Deprecated Or Bad Checkpoints

- `checkpoints/alignn_mt_inorg` should not be used as a standalone formal
  ALIGNN model. The latest MACE+ALIGNN result should be reported from the
  archived fusion checkpoint
  `checkpoints/dual_backbone_inorg_preAlignSplit_20260702_080248`.
- `checkpoints/dual_backbone_inorg_oldalignn_20260705_072747` is retained as a
  provenance-clear recovered-ALIGNN rerun. It is not the latest reporting
  checkpoint because its BG MAE is higher than the archived best-BG checkpoint.
- `checkpoints/mace_m3gnet_inorg` is the bad current MACE+M3GNet dual run and
  should only be kept for comparison or debugging. It is not the latest formal
  MACE+M3GNet model.

## Reference Comparison

| Model/run | BG MAE | GT F1 | EH F1 | Status |
|---|---:|---:|---:|---|
| MACE+ALIGNN archived best-BG checkpoint | 0.2218 | 0.8476 | 0.9533 | Latest reporting |
| MACE+ALIGNN recovered old ALIGNN | 0.2366 | 0.8566 | 0.9495 | Provenance-clear rerun |
| MACE+M3GNet recovered old M3GNet | 0.2587 | 0.8537 | 0.9576 | Latest official |
| MACE+M3GNet current bad run | 0.4852 | 0.6055 | 0.8321 | Deprecated/debug only |

The recovery results show that the dual-fusion code path is valid. The degraded
runs were caused by newly trained backbone checkpoints whose hidden
representations did not transfer well into the fusion models.
