# MechaGRN

A deep learning framework for Gene Regulatory Network (GRN) inference from single-cell RNA sequencing data.

## Overview

MechaGRN is a mechanism-aware graph neural network designed to infer gene regulatory networks from scRNA-seq expression data. The model incorporates:

- **Direction-aware graph attention** for capturing TF→Target regulatory relationships
- **High-order structural perception** for modeling indirect regulatory effects
- **Expression-aware encoding** with sparsity gating for single-cell data characteristics

## Project Structure

```
MechaGRN/
├── Code/
│   ├── main.py           # Training script
│   ├── Model.py          # Neural network models
│   ├── preprocessing.py  # Data loading and preprocessing
│   ├── DatasetsSplit.py  # Dataset splitting utilities
│   └── Tools.py          # Evaluation metrics and utilities
```

## Model Architecture

### Core Components

| Component | Description |
|-----------|-------------|
| `FocalLoss` | Handles class imbalance in GRN prediction |
| `RegulationAwareExpressionTransformer` | Encodes gene expression with statistical features and sparsity-aware gating |
| `DirectionAwareGraphLayer` | Bidirectional attention for TF→Target and Target→TF relationships |
| `HighOrderPerceptionLayer` | Captures second-order regulatory interactions |
| `RegulationAwareDecoder` | Edge prediction via feature concatenation |
| `MechaGRN` | Main model integrating all components |

### Architecture Flow

```
Expression Data → Expression Encoder → Direction-Aware Graph Layers → High-Order Enhancement → Decoder → Regulatory Links
```

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- SciPy
- tqdm

## Dataset Format

The model expects benchmark datasets organized as:

```
Dataset/
├── Benchmark Dataset/
│   └── {net_type} Dataset/
│       └── {cell_type}/
│           └── TFs+{gene_num}/
│               ├── BL--ExpressionData.csv   # Gene expression matrix
│               ├── TF.csv                   # Transcription factor indices
│               └── Target.csv               # Target gene indices
│               └── Label.csv                # Ground truth regulatory links
├── train/
├── val/
└── test/
```

## Usage

### 1. Split Dataset

```bash
python DatasetsSplit.py
```

Configure `net_types`, `data_types`, and `gene_num` in the script.

### 2. Train Model

```bash
python main.py --lr 3e-4 --epochs 100 --hidden_dim 32 --batch_size 256
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 3e-4 | Learning rate |
| `--epochs` | 100 | Training epochs |
| `--hidden_dim` | 32 | Hidden dimension |
| `--output_dim` | 16 | Output dimension |
| `--num_heads` | 4 | Attention heads |
| `--batch_size` | 256 | Batch size |
| `--seed` | 40 | Random seed |

## Evaluation Metrics

The model evaluates using:

- **AUROC** - Area Under ROC Curve
- **AUPRC** - Area Under Precision-Recall Curve
- **F1 Score** - Harmonic mean of precision and recall
- **Early Precision (EP)** - Precision in early recall region (≤0.1)

## Supported Benchmark Datasets

- **Non-Specific** - General regulatory networks
- **Specific** - Cell-type specific networks
- **STRING** - Protein-protein interaction based
- **Lofgof** - Knockout perturbation data

Cell types: hESC, hHEP, mDC, mESC, mHSC-E, mHSC-GM, mHSC-L

## Output

- Best model saved to `model/{net_type}/{cell_type} {gene_num}/best_model.pkl`
- Results logged to `results.txt`

## License

This project is for academic research purposes.