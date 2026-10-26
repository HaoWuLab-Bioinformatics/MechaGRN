# MechaGRN

A deep learning framework for **Gene Regulatory Network (GRN) inference** from single-cell RNA sequencing (scRNA-seq) data.

## Overview

MechaGRN is a mechanism-aware graph neural network designed to infer gene regulatory networks from scRNA-seq expression data. It formulates GRN inference as a **link-prediction** task over a gene graph: given an expression matrix and a set of known regulatory edges, the model learns to predict whether a directed link (TF → Target) exists between any TF–gene pair.

The model incorporates:

- **Direction-aware graph attention** — separately models forward (TF → Target) and backward (Target → TF) information flow to capture the directionality of regulation.
- **High-order structural perception** — propagates over the 2-hop (second-order) adjacency to model indirect regulatory effects.
- **Expression-aware encoding with sparsity gating** — encodes raw expression together with statistical features (mean, variance, dropout rate) and gates them to handle the high sparsity of single-cell data.
- **Focal Loss** — addresses the severe positive/negative class imbalance typical of GRN datasets.

## Directional formulation (Eq. 12–13)

The graph fed to the two direction-aware attention branches is the **directed**
TF → target skeleton of the training links, `A[tf, target] = 1`. The two
branches are

- forward: `H_forward = Attn(A, X)` — a target aggregates over its incoming TFs;
- backward: `H_backward = Attn(Aᵀ, X)` — a TF aggregates over its targets;

combined with layer-specific, learnable weights constrained to a probability
simplex,

```
[α_ℓ, β_ℓ] = softmax([a_ℓ, b_ℓ])          α_ℓ + β_ℓ = 1,  α_ℓ, β_ℓ > 0
H_dir      = α_ℓ · H_forward + β_ℓ · H_backward
```

Each `DirectionAwareGraphLayer` owns its own `dir_logits = [a_ℓ, b_ℓ]`, so the
two layers may weight the two directions differently. The logits are
initialized at `[log 0.9, log 0.1]`, so training starts from
`(α_ℓ, β_ℓ) = (0.9, 0.1)` and the split between the two directions is then
learned end-to-end from the data.

After training the learned coefficients are printed for the record:

```
[DirectionWeights] graph_layer1: alpha_l=... beta_l=... (sum=1.0000)
[ResidualWeight]   graph_layer1: wolf=...
[DirectionWeights] graph_layer2: alpha_l=... beta_l=... (sum=1.0000)
[ResidualWeight]   graph_layer2: wolf=...
[HighOrderWeight]  high_order: lambda_ho=...
```

The high-order coefficient of Eq. 17 is stored as `lambda_ho` in the code, so
that it is not confused with the focal-loss focusing parameter γ of Eq. 21.

### Configuration and reproducibility

The directional coefficients `α_ℓ, β_ℓ`, the residual coefficient `wolf` and the
high-order coefficient `lambda_ho` are trainable parameters, optimized together
with the rest of the network. The remaining choices are exposed as command-line
flags rather than hard-coded values, and each run prints both its resolved
configuration and the fitted coefficients, so a run can be reconstructed from
its log alone.

| Flag / item | Behavior |
|---|---|
| `--direction` | `True` (default) builds the directed TF → target graph of Eq. 12; `False` builds the symmetric undirected graph, used as an ablation |
| `--loop` | `False` (default) leaves self-loops out of the adjacency matrix |
| `--focal_alpha`, `--focal_gamma` | the two focal-loss parameters of Eq. 21 (defaults `0.75` and `2.0`, as used for all benchmark runs) |
| `--seed` | fixes the `random`, `numpy`, `torch` and CUDA seeds |
| Boolean flags | parsed by `str2bool`, so `--direction False` is read as `False` |
| Checkpoint keys | `…dir_logits` and `…high_order.lambda_ho`; `load_state_dict_compat` additionally maps archives that store `…alpha` / `…beta` / `high_order.gamma`, so saved `best_model.pkl` files remain loadable |

`Code/tests/test_direction_adjacency.py` checks the adjacency orientation, the
handling of negative training rows and the softmax-normalized directional
coefficients. It deliberately does not import `torch` or `scipy`, so it runs on
a bare Python + NumPy installation:

```bash
python Code/tests/test_direction_adjacency.py
```

A typical run from the `Code/` directory:

```bash
python main.py --direction True --loop False --focal_alpha 0.75 --focal_gamma 2.0 --seed 40
```

and the configuration line it reports:

```
[Config] directed_adjacency=True | self_loops=False | edges=... | adjacency_is_symmetric=False | focal_alpha=0.75 | focal_gamma=2.0 | seed=40
```

## Project Structure

```
MechaGRN/
├── Code/
│   ├── main.py            # Training / evaluation entry point
│   ├── Model.py           # Neural network models (MechaGRN + submodules)
│   ├── preprocessing.py   # Data loading, normalization, adjacency building, network stats
│   ├── DatasetsSplit.py   # Train/validation/test split utilities
│   ├── Tools.py           # Evaluation metrics, best-model saving, logging
│   └── tests/
│       └── test_direction_adjacency.py  # Edge-direction regression tests
├── Dataset/
│   ├── Benchmark Dataset/  # Raw benchmark data (expression + ground-truth network)
│   ├── train/              # Pre-split training edge sets
│   ├── val/                # Pre-split validation edge sets
│   └── test/               # Pre-split test edge sets
└── README.md
```

## Example Datasets (for quick testing)

To let readers **reproduce and test the pipeline immediately without downloading external data**, this repository ships with a ready-to-run example: the **STRING** benchmark for the **hESC** (human embryonic stem cell) cell type, at two network scales (`TFs+500` and `TFs+1000`). Both the raw benchmark data and the pre-computed train/validation/test splits are included, so you can run training directly.

### What is included

```
Dataset/
├── Benchmark Dataset/
│   └── STRING Dataset/
│       └── hESC/
│           ├── TFs+500/
│           │   ├── BL--ExpressionData.csv   # Expression matrix (910 genes × 758 cells)
│           │   ├── BL--network.csv          # Raw regulatory network
│           │   ├── Label.csv                # Ground-truth links (4,257 edges)
│           │   ├── TF.csv                    # 411 transcription factors + indices
│           │   └── Target.csv               # Candidate target genes + indices
│           └── TFs+1000/
│               ├── BL--ExpressionData.csv   # Expression matrix (1,410 genes × 758 cells)
│               ├── BL--network.csv
│               ├── Label.csv                # Ground-truth links (5,149 edges)
│               ├── TF.csv
│               └── Target.csv
├── train/STRING/
│   ├── hESC 500/Train_set.csv               # 4,636 labeled edges
│   └── hESC 1000/Train_set.csv              # 5,582 labeled edges
├── val/STRING/
│   ├── hESC 500/Validation_set.csv          # 830 labeled edges
│   └── hESC 1000/Validation_set.csv         # 1,080 labeled edges
└── test/STRING/
    ├── hESC 500/Test_set.csv                # 63,499 labeled edges
    └── hESC 1000/Test_set.csv               # 86,571 labeled edges
```

### Summary of example data

| Scale | Genes | Cells | Ground-truth links | TFs | Train | Validation | Test |
|-------|------:|------:|-------------------:|----:|------:|-----------:|-----:|
| `TFs+500`  |   910 | 758 | 4,257 | 411 | 4,636 |   830 | 63,499 |
| `TFs+1000` | 1,410 | 758 | 5,149 | 411 | 5,582 | 1,080 | 86,571 |

> **Note:** The train/val/test splits are already provided for the example data, so you can skip the splitting step and jump straight to training. If you want to regenerate the splits yourself, see [Advanced: Regenerating splits](#advanced-regenerating-splits).

## Data Format

The model expects benchmark datasets organized as follows:

```
Dataset/
├── Benchmark Dataset/
│   └── {net_type} Dataset/
│       └── {cell_type}/
│           └── TFs+{gene_num}/
│               ├── BL--ExpressionData.csv   # Gene expression matrix
│               ├── TF.csv                   # Transcription factor indices
│               ├── Target.csv               # Candidate target gene indices
│               └── Label.csv                # Ground-truth regulatory links
├── train/{net_type}/{cell_type} {gene_num}/Train_set.csv
├── val/{net_type}/{cell_type} {gene_num}/Validation_set.csv
└── test/{net_type}/{cell_type} {gene_num}/Test_set.csv
```

### File descriptions

| File | Columns | Description |
|------|---------|-------------|
| `BL--ExpressionData.csv` | gene name + one column per cell | Normalized/log expression matrix (genes × cells). The first column is the gene name; each remaining column is a single cell. |
| `TF.csv` | `TF`, `index` | Transcription factor gene names and their integer row index in the expression matrix. |
| `Target.csv` | `Gene`, `index` | Candidate target gene names and their integer index. |
| `Label.csv` | `TF`, `Target` | Ground-truth regulatory links as integer index pairs (positive edges). |
| `Train_set.csv` / `Validation_set.csv` / `Test_set.csv` | `TF`, `Target`, `Label` | Labeled edge samples where `Label` is `1` (regulatory link) or `0` (negative). |

## Requirements

- Python 3.x
- PyTorch (CUDA optional — the code automatically falls back to CPU)
- NumPy
- Pandas
- Scikit-learn
- SciPy
- tqdm

Install dependencies:

```bash
pip install torch numpy pandas scikit-learn scipy tqdm
```

## Quick Start — Run the Example Dataset

The included example data is **STRING / hESC**, so before training you need to point `main.py` at it. Open [`Code/main.py`](Code/main.py) and set the experiment configuration in the `__main__` block:

```python
# Code/main.py (bottom of the file)
net_types = ["STRING"]      # use the included example benchmark
cell_types = ["hESC"]       # use the included cell type
gene_num = 500              # 500 or 1000 (both example scales are provided)
```

Then run from the `Code/` directory:

```bash
cd Code
python main.py --lr 3e-4 --epochs 100 --hidden_dim 32 --batch_size 256
```

The script will:

1. Load and normalize the expression matrix.
2. Build a sparse adjacency matrix from the training edges.
3. Train MechaGRN, evaluating on the validation set every epoch and saving the best checkpoint.
4. Load the best checkpoint and report final metrics on the test set.
5. Append the results to `results.txt`.

To try the larger network, set `gene_num = 1000` and re-run.

## Usage

### 1. Train / Evaluate

```bash
cd Code
python main.py --lr 3e-4 --epochs 100 --hidden_dim 32 --batch_size 256
```

The `net_type`, `cell_type`, and `gene_num` are configured directly in the `__main__` block of `main.py` (see [Quick Start](#quick-start--run-the-example-dataset)).

#### Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 3e-4 | Learning rate |
| `--epochs` | 100 | Number of training epochs |
| `--hidden_dim` | 32 | Hidden dimension size |
| `--output_dim` | 16 | Output embedding dimension |
| `--num_heads` | 4 | Number of attention heads |
| `--batch_size` | 256 | Training batch size |
| `--direction` | True | Build a directed TF → target adjacency matrix (Eq. 12); `False` builds the undirected graph used for the ablation |
| `--loop` | False | Whether to add self-loops to the adjacency matrix |
| `--focal_alpha` | 0.75 | Focal-loss positive-class weight (Eq. 21) |
| `--focal_gamma` | 2.0 | Focal-loss focusing parameter (Eq. 21) |
| `--seed` | 40 | Random seed for reproducibility |

### 2. Advanced: Regenerating splits

The example data already contains train/val/test splits. If you add your own benchmark data (or want to re-shuffle the splits), use the splitting utility. Configure `net_types`, `data_types`, and `gene_num` in the `__main__` block of [`Code/DatasetsSplit.py`](Code/DatasetsSplit.py), then run:

```bash
cd Code
python DatasetsSplit.py --ratio 0.67 --p_val 0.5
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--ratio` | 0.67 | Fraction of positive links assigned to the training set |
| `--p_val` | 0.5 | Probability that a single-target TF's link goes to the training set |

- For **Specific** networks, hard-negative sampling is used (negatives are genes *not* regulated by the TF).
- For all other network types, negatives are drawn by random sampling calibrated to each network's density.

## Model Architecture

### Core Components

| Component | Description |
|-----------|-------------|
| `FocalLoss` | Focal loss to handle severe class imbalance in GRN prediction |
| `RegulationAwareExpressionTransformer` | Encodes expression with statistical features and sparsity-aware gating, refined by self-attention |
| `DirectionAwareGraphLayer` | Bidirectional attention modeling TF→Target and Target→TF relationships |
| `HighOrderPerceptionLayer` | Captures second-order (2-hop) regulatory interactions |
| `RegulationAwareDecoder` | Edge prediction from concatenated TF/Target features (`[h_i, h_j, h_i−h_j, h_i·h_j]`) |
| `MechaGRN` | Main model integrating all components |

### Architecture Flow

```
Expression Data
      │
      ▼
Expression Encoder (sparsity-gated + self-attention)
      │
      ▼
Direction-Aware Graph Layers (×2)
      │
      ▼
High-Order Perception (2-hop enhancement)
      │
      ▼
TF / Target Projections → Decoder → Regulatory Link Score
```

## Evaluation Metrics

The model is evaluated using:

- **AUROC** — Area Under the ROC Curve
- **AUPRC** — Area Under the Precision-Recall Curve
- **F1 Score** — Harmonic mean of precision and recall (at threshold 0.5)
- **Early Precision (EP)** — Mean precision in the early recall region (recall ≤ 0.1), important for ranking true links first in GRN inference

The best model is selected on the validation set using a combined score of `0.2 × AUROC + 0.8 × AUPRC`.

## Supported Benchmark Datasets

While the repository ships with the **STRING / hESC** example, the pipeline supports the following benchmark network types:

- **Non-Specific** — General regulatory networks
- **Specific** — Cell-type specific networks (uses hard-negative sampling)
- **STRING** — Protein-protein interaction based (included example)
- **Lofgof** — Loss/gain-of-function knockout perturbation data

Supported cell types: `hESC`, `hHEP`, `mDC`, `mESC`, `mHSC-E`, `mHSC-GM`, `mHSC-L`
(Network scales: 500 or 1000 genes. `Lofgof` is only available for `mESC`.)

## Output

- **Best model** saved to `Code/model/{net_type}/{cell_type} {gene_num}/best_model.pkl`
- **Results log** appended to `Code/results.txt` (network type, cell type, gene number, AUROC, AUPRC, F1)

## License

This project is for academic research purposes.
