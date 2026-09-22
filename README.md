#  Prior-Enhanced Progressive Multi-Relation Learning for Accurate Cancer Drug Response Prediction

This project implements a PyTorch regression model that combines molecular structure, target protein embeddings, cell-line multi-omics features, and four bipartite biological relations to predict drug–cell-line responses.

## 1. Project files

| File | Purpose |
| --- | --- |
| `model.py` | D-MPNN encoder, multi-head bipartite attention, gated drug fusion, multi-omics encoders, and MLP regressor. |
| `train.py` | CSV loading, molecular graph construction, entity alignment, splitting, preprocessing, training, evaluation, and attention export. |

Keep the two Python files in the same directory. Training imports `MultiViewModel` from `model1_new`. The input datasets must be supplied separately.

## 2. Model

### Node features

- **Drugs:** Drug SMILES.
- **Targets:** precomputed ESM-2 embeddings loaded from a CSV file. ESM-2 itself is not loaded or fine-tuned by these scripts.
- **Pathways:** a trainable `nn.Embedding` indexed by pathway IDs.
- **Cell lines:** mutation, copy-number variation (CNV), and RNA features.

### Relations and propagation

| Relation               | Implemented propagation               | Edge prior    |
| ---------------------- | ------------------------------------- | ------------- |
| Pathway–target (PT)    | Target → pathway and pathway → target | Topology only |
| Drug–target (DT)       | Target → drug                         | Topology only |
| Drug–pathway (DP)      | Pathway → drug                        | Input `w_dp`  |
| Pathway–cell line (PC) | Pathway → cell line                   | ssGSEA score  |

## 3. Dependencies

Direct third-party dependencies:

```text
- Please install the environment using anaconda3;
  conda create -n PCGraph-DRP python=3.11.11
- Install the necessary packages.
  torch == 2.6.0+cu118
  torch_cluster == 1.6.3+pt26cu118
  torch-geometric == 2.6.1
  torch_scatter == 2.1.2+pt26cu118
  torch_sparse == 0.6.18+pt26cu118
  torch_spline_conv == 1.2.2+pt26cu118
  torchaudio == 2.6.0+cu118
  torchvision == 0.21.0+cu118
  tornado == 6.5.1
  numpy  == 1.26.4
  pandas == 1.5.3
  scipy == 1.13.1
  rdkit == 2025.3.2
```

## 4. Input data

Set the following constants at the top of `train1_new.py`. All inputs are CSV files.

| Constant | Expected format |
| --- | --- |
| `DRUG_SMILES_PATH` | `drug_name` and `smiles`; |
| `IC50_PATH` | Wide response matrix; |
| `TARGET_ESM2_PATH` | `protein` plus numeric embedding columns; the identifier `portein` is also accepted. |
| `OMICS_PATH` | `cell_line` (or the first column), with numeric features ending in `_mutation`, `_cnv`, and `_rna`. |
| `SSGSEA_PATH` | `cell_line` plus pathway columns containing numeric ssGSEA scores. |
| `DT_WIDE_PATH` | `drug_name` plus columns containing target names as values, not binary indicators. |
| `DP_EDGE_PATH` | `drug_name`, `pathway`, and numeric `w_dp`. |
| `PATHWAY_GENE_MEMBERSHIP_PATH` | `pathway` and `genes`, with multiple genes separated by a pipe character. |

## 5. Run

1. Place `model.py` and `train.py` in the same directory.
2. Prepare the input CSV files described above.
3. Replace all input path constants and `OUT_DIR` in `train1.py` with your local paths. The supplied paths point to the original author's filesystem.
4. Run:

```bash
python train.py
```

Configuration is edited directly in the script; no command-line argument parser is provided. The output directory is created automatically. Reusing the same output directory overwrites the corresponding output files.

Each epoch prints training loss, validation metrics, and the mean drug fusion gate. After training, the best validation checkpoint is reloaded and evaluated on the test split.

## 6. Evaluation and outputs

Reported metrics are MSE, MAE, RMSE, R², Pearson correlation (PCC), and Spearman correlation (SCC). Predictions and labels are inverse-transformed before evaluation. Training loss and reported evaluation errors therefore use different scales.

| Output in `OUT_DIR`                | Contents                                                     |
| ---------------------------------- | ------------------------------------------------------------ |
| `best_model_moe_gate.pt`           | Model `state_dict` selected by validation RMSE.              |
| `y_scaler_moe_gate.pkl`            | Fitted response `StandardScaler`.                            |
| `attn_maps_moe_gate_best_test.pkl` | Edge indices, attention arrays, and drug gates from final evaluation; written when `SAVE_ATTN=True`. |
