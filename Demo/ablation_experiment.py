import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# When this script lives under Demo/, make sure we import the implementation
# from the Code/ directory (otherwise Demo/Model.py might be used by accident).
CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from Model import MechaGRN
from preprocessing import scRNADataset, load_data, adj2saprse_tensor


def _resolve_first_existing(paths: List[str]) -> str:
    """Return the first path that exists, otherwise raise FileNotFoundError."""
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these files exist: {paths}")


def compute_precision_recall_at_k(
    pred_df: pd.DataFrame,
    true_set: set,
    k_values: list,
    tf_col: str = "tf",
    target_col: str = "target",
    score_col: str = "pred_score",
    deduplicate: bool = True,
    score_agg: str = "max",
) -> pd.DataFrame:
    """
    Compute global Precision@K and Recall@K for GRN edge prediction.

    Precision@K = (Top-K predicted edges that are in true_set) / K
    Recall@K    = (Top-K predicted edges that are in true_set) / |true_set|

    Notes:
    - Global Top-K: sort all candidate edges across all TF-target pairs by pred_score.
    - true_set elements must be tuples like (tf, target) matching pred_df types exactly.
    """
    if len(true_set) == 0:
        raise ValueError("true_set is empty, so Recall@K denominator would be 0.")

    if not isinstance(pred_df, pd.DataFrame):
        raise TypeError("pred_df must be a pandas DataFrame.")

    needed_cols = [tf_col, target_col, score_col]
    missing = [c for c in needed_cols if c not in pred_df.columns]
    if missing:
        raise ValueError(f"pred_df is missing required columns: {missing}")

    # Validate/normalize k_values
    k_values = [int(k) for k in k_values]
    if any(k <= 0 for k in k_values):
        raise ValueError("All k_values must be positive integers.")

    # Clean rows
    df = pred_df.loc[:, needed_cols].copy()
    df = df.dropna(subset=needed_cols)
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[score_col])

    # Optional deduplication: keep one score per edge (tf,target)
    if deduplicate:
        df["_edge_key"] = list(zip(df[tf_col].values, df[target_col].values))

        if score_agg not in ("max", "min", "mean", "median"):
            raise ValueError("score_agg must be one of: 'max', 'min', 'mean', 'median'")

        df = (
            df.groupby("_edge_key", as_index=False)[score_col]
              .agg(score_agg)
              .rename(columns={score_col: score_col})
        )
        df[[tf_col, target_col]] = pd.DataFrame(df["_edge_key"].tolist(), index=df.index)
        df = df.drop(columns=["_edge_key"])

    # Global sorting by score (descending)
    df = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)

    # Compute TP flags for each sorted prediction
    pred_keys: List[Tuple] = list(zip(df[tf_col].values, df[target_col].values))
    tp_flags = np.fromiter((key in true_set for key in pred_keys), dtype=np.int64)
    cum_tp = np.cumsum(tp_flags)  # cum_tp[i] = TP count in top (i+1)

    n = len(df)
    out_rows = []
    for K in sorted(k_values):
        K_used = min(K, n)
        tp_at_k = int(cum_tp[K_used - 1]) if K_used > 0 else 0
        precision_at_k = tp_at_k / K_used if K_used > 0 else 0.0
        recall_at_k = tp_at_k / len(true_set)
        out_rows.append(
            {
                "K": K,
                "K_used": K_used,
                "tp": tp_at_k,
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
            }
        )

    return pd.DataFrame(out_rows)


def evaluate_one_ablation(
    *,
    net_type: str,
    cell_type: str,
    gene_num: int,
    hidden_dim: int,
    output_dim: int,
    num_heads: int,
    use_expression: bool,
    use_D: bool,
    use_H: bool,
    use_E: bool,
    k_values: List[int],
    checkpoint_dir: str,
    device: torch.device,
) -> pd.DataFrame:
    """
    Evaluate one ablation configuration on test_set.csv:
    - pred_df from model logits (sigmoid score in [0,1])
    - true_set from Label==1 edges in test_set.csv
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
    dataset_dir = os.path.join(root_dir, "Dataset")

    # Build dataset file paths (handles potential case differences)
    exp_file = os.path.join(
        dataset_dir,
        "Benchmark Dataset",
        f"{net_type} Dataset",
        cell_type,
        f"TFs+{gene_num}",
        "BL--ExpressionData.csv",
    )
    tf_file = os.path.join(
        dataset_dir,
        "Benchmark Dataset",
        f"{net_type} Dataset",
        cell_type,
        f"TFs+{gene_num}",
        "TF.csv",
    )

    base_split_dir = os.path.join(dataset_dir, "test", net_type, f"{cell_type} {gene_num}")
    test_file = _resolve_first_existing(
        [
            os.path.join(base_split_dir, "test_set.csv"),
            os.path.join(base_split_dir, "Test_set.csv"),
        ]
    )

    base_split_dir_train = os.path.join(dataset_dir, "train", net_type, f"{cell_type} {gene_num}")
    train_file = _resolve_first_existing(
        [
            os.path.join(base_split_dir_train, "Train_set.csv"),
        ]
    )

    base_split_dir_val = os.path.join(dataset_dir, "val", net_type, f"{cell_type} {gene_num}")
    val_file = _resolve_first_existing(
        [
            os.path.join(base_split_dir_val, "Validation_set.csv"),
            os.path.join(base_split_dir_val, "validation_set.csv"),
        ]
    )

    # 1) Load expression features + TF indices
    data_input = pd.read_csv(exp_file, index_col=0)
    loader = load_data(data_input)
    feature = loader.exp_data()
    feature_t = torch.tensor(feature, dtype=torch.float32).to(device)

    tf = pd.read_csv(tf_file, index_col=0)["index"].values.astype(np.int64)
    tf_t = torch.tensor(tf, device=device)

    # 2) Build adjacency from training set positives (same as training code)
    train_df = pd.read_csv(train_file, index_col=0)
    train_data = train_df.values
    train_load = scRNADataset(train_data, feature.shape[0])
    adj = train_load.Adj_Generate(tf_t, loop=False)
    adj = adj2saprse_tensor(adj).to(device)

    # 3) Load test candidate edges + labels
    test_df = pd.read_csv(test_file, index_col=0)
    # Important: TF/Target columns are used as indices inside the model.
    # Therefore the tensor must be an integer type.
    test_data = torch.tensor(test_df.values, dtype=torch.long).to(device)

    # 4) Build model and load checkpoint weights
    model = MechaGRN(
        input_dim=feature.shape[1],  # gene expression matrix feature dim
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        device=str(device),
        use_expression=use_expression,
    ).to(device)

    state_dict = torch.load(os.path.join(checkpoint_dir, "best_model.pkl"), map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    def predict_logits_for_test_edges(
        model: MechaGRN,
        feature_t: torch.Tensor,
        adj_t: torch.Tensor,
        test_sample_t: torch.Tensor,
        *,
        use_expression: bool,
        use_D: bool,
        use_H: bool,
        use_E: bool,
    ) -> torch.Tensor:
        """
        Re-implement MechaGRN forward with ablation toggles.

        Important:
        - Keep tensor shapes consistent with the full decoder (edge_feat dim = output_dim*4).
        - For w/o E: zero out (h_i - h_j) and (h_i * h_j) parts, so we don't need to change decoder parameters.
        """
        # ===== Expression / fallback encoding =====
        if use_expression:
            node_embed = model.expr_encoder(feature_t)
        else:
            node_embed = model.fallback_proj(feature_t)

        # (1, G, d)
        node_embed = node_embed.unsqueeze(0)

        # ===== D: direction-aware propagation =====
        if use_D:
            node_embed = model.graph_layer1(node_embed, adj_t)
            node_embed = model.graph_layer2(node_embed, adj_t)

        # ===== H: high-order structural enhancement =====
        if use_H:
            node_embed = model.high_order(node_embed, adj_t)

        node_embed = node_embed.squeeze(0)  # (G, d)

        # ===== node embeddings for TF/Target =====
        tf_embed = F.elu(model.tf_linear(node_embed))
        target_embed = F.elu(model.target_linear(node_embed))

        tf_idx = test_sample_t[:, 0].long()
        target_idx = test_sample_t[:, 1].long()

        train_tf = tf_embed[tf_idx]
        train_target = target_embed[target_idx]

        # ===== E: enhanced edge features =====
        if use_E:
            edge_feat = torch.cat(
                [train_tf, train_target, train_tf - train_target, train_tf * train_target],
                dim=1,
            )
        else:
            zeros = torch.zeros_like(train_tf)
            edge_feat = torch.cat([train_tf, train_target, zeros, zeros], dim=1)

        pred = model.decoder.mlp(edge_feat)
        pred = torch.nan_to_num(pred)
        return pred

    # 5) Predict scores for all candidate edges in test_set.csv
    with torch.no_grad():
        logits = predict_logits_for_test_edges(
            model,
            feature_t,
            adj,
            test_data,
            use_expression=use_expression,
            use_D=use_D,
            use_H=use_H,
            use_E=use_E,
        )
        # Keep the same score post-processing as Code/main.py during testing
        score_test = logits / 0.8
        score_test = torch.clamp(score_test, -6, 6)
        score_test = torch.sigmoid(score_test)

    pred_scores = score_test.detach().cpu().numpy().reshape(-1)

    pred_df = pd.DataFrame(
        {
            "tf": test_df["TF"].values,
            "target": test_df["Target"].values,
            "pred_score": pred_scores,
        }
    )

    # true_set: all positive edges inside the candidate set
    pos_mask = test_df["Label"].values.astype(int) == 1
    true_set = set(zip(test_df.loc[pos_mask, "TF"].values, test_df.loc[pos_mask, "Target"].values))

    # 6) Compute global Precision@K and Recall@K
    metrics = compute_precision_recall_at_k(
        pred_df=pred_df,
        true_set=true_set,
        k_values=k_values,
        tf_col="tf",
        target_col="target",
        score_col="pred_score",
        deduplicate=True,
        score_agg="max",
    )

    return metrics


def get_module_ablation_configs() -> Dict[str, Dict[str, bool]]:
    """
    Map ablation name -> {use_D, use_H, use_E}.

    Definitions (as requested by user):
    - full_model: include D, H, E
    - w/o D: remove D, keep H and E
    - w/o H: remove H, keep D and E
    - w/o E: remove E, keep D and H
    - w/o M: remove D and H, keep E
    """
    return {
        "full_model": {"use_D": True, "use_H": True, "use_E": True},
        "w/o D": {"use_D": False, "use_H": True, "use_E": True},
        "w/o H": {"use_D": True, "use_H": False, "use_E": True},
        "w/o E": {"use_D": True, "use_H": True, "use_E": False},
        "w/o M": {"use_D": False, "use_H": False, "use_E": True},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_type", type=str, default="Specific")
    parser.add_argument("--cell_type", type=str, default="mHSC-GM")
    parser.add_argument("--gene_num", type=int, default=500)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--output_dim", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--k_values", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--use_expression", type=int, default=1, help="1 use expression encoder, 0 fallback proj")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="Override checkpoint dir. Default uses ./Code/model/<net_type>/<cell_type> <gene_num>/",
    )
    parser.add_argument("--out_csv", type=str, default="module_ablation_pr_recall.csv")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if args.checkpoint_dir:
        ckpt_dir = args.checkpoint_dir
    else:
        ckpt_dir = os.path.join(
            root_dir,
            "Code",
            "model",
            args.net_type,
            f"{args.cell_type} {args.gene_num}",
        )

    os.makedirs(os.path.dirname(os.path.join(root_dir, "Code", args.out_csv)) or ".", exist_ok=True)

    use_expression = bool(int(args.use_expression))
    module_cfgs = get_module_ablation_configs()

    results = []
    for ablation_name, flags in module_cfgs.items():
        df = evaluate_one_ablation(
            net_type=args.net_type,
            cell_type=args.cell_type,
            gene_num=args.gene_num,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_heads=args.num_heads,
            use_expression=use_expression,
            use_D=flags["use_D"],
            use_H=flags["use_H"],
            use_E=flags["use_E"],
            k_values=args.k_values,
            checkpoint_dir=ckpt_dir,
            device=device,
        )
        df.insert(0, "ablation", ablation_name)
        df.insert(1, "cell_type", args.cell_type)
        df.insert(2, "net_type", args.net_type)
        df.insert(3, "gene_num", args.gene_num)
        results.append(df)

    all_df = pd.concat(results, ignore_index=True)

    out_path = os.path.join(root_dir, "Code", args.out_csv)
    all_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved ablation metrics to: {out_path}")
    print(all_df)


if __name__ == "__main__":
    main()

