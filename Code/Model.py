import torch
import torch.nn as nn
import torch.nn.functional as F

# Disable cuDNN benchmark for reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


# =====================================================
# 1️⃣ Focal Loss - Handles class imbalance in GRN prediction
# =====================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in binary classification.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for positive class (default: 0.75)
        gamma: Focusing parameter to reduce loss on well-classified samples (default: 2.0)
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # Compute binary cross entropy without reduction
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Calculate probability and pt (true class probability)
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1 - prob) * (1 - targets)

        # Apply focal weighting
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


# =====================================================
# 2️⃣ Expression Encoder - Transforms gene expression into embeddings
# =====================================================

class RegulationAwareExpressionTransformer(nn.Module):
    """
    Expression encoder with sparsity-aware gating mechanism.
    Captures both expression patterns and statistical features (mean, variance, dropout rate).

    Args:
        input_dim: Number of cells (expression matrix columns)
        hidden_dim: Dimension of hidden representations
        num_heads: Number of attention heads for self-attention
        dropout: Dropout rate for regularization
    """

    def __init__(self, input_dim, hidden_dim=64, num_heads=4, dropout=0.1):
        super().__init__()

        # Expression value projection
        self.expr_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # Statistical features projection (mean, variance, dropout rate)
        self.stat_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # Sparsity-aware gating network
        # Controls information flow based on expression sparsity
        self.sparsity_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )

        # ===== Self-Attention Layer =====
        # Refines node embeddings through inter-gene attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Feed-forward network (currently disabled)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Forward pass for expression encoding.

        Args:
            x: Gene expression matrix (num_genes, num_cells)

        Returns:
            h: Node embeddings (num_genes, hidden_dim)
        """

        # Expression projection
        expr_embed = self.expr_proj(x)

        # Compute statistical features
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True)
        dropout_rate = torch.mean((x == 0).float(), dim=1, keepdim=True)

        stats = torch.cat([mean, var, dropout_rate], dim=1)
        stat_embed = self.stat_proj(stats)

        # Sparsity-aware gating
        # Gates expression embeddings based on sparsity patterns
        gate = self.sparsity_gate(x)
        expr_embed = expr_embed * gate

        # Feature fusion
        fusion_input = torch.cat([expr_embed, stat_embed], dim=1)
        h = self.fusion(fusion_input)

        # ===== Self-attention refinement =====
        h = h.unsqueeze(0)  # Add batch dimension: (1, N, hidden_dim)

        attn_out, _ = self.attn(h, h, h)
        h = self.attn_norm(h + attn_out)

        # Feed-forward network (currently disabled)
        # ffn_out = self.ffn(h)
        # h = self.ffn_norm(h + ffn_out)

        h = h.squeeze(0)  # Remove batch dimension: (N, hidden_dim)

        return h


# =====================================================
# 3️⃣ Direction-Aware Graph Transformer Layer
# Captures bidirectional regulatory relationships (TF→Target and Target→TF)
# =====================================================

class DirectionAwareGraphLayer(nn.Module):
    """
    Graph attention layer with directional awareness.
    Separates forward (TF→Target) and backward (Target→TF) information flow.

    Args:
        embed_dim: Dimension of node embeddings
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()

        # Forward attention: TF regulates Target
        self.attn_forward = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Backward attention: Target influenced by TF
        self.attn_backward = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Weighting factors for directional information
        self.alpha = nn.Parameter(torch.tensor(0.9), requires_grad=False)  # Forward weight
        self.beta = nn.Parameter(torch.tensor(0.1), requires_grad=False)   # Backward weight
        self.wolf = nn.Parameter(torch.tensor(0.01), requires_grad=False)  # Residual weight

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        Forward pass with directional graph attention.

        Args:
            x: Node embeddings (1, num_genes, embed_dim)
            adj: Sparse adjacency matrix

        Returns:
            Updated node embeddings
        """

        B, G, d = x.size()
        adj_dense = adj.to_dense()

        # Forward mask (TF → Target)
        # Includes self-loops for stability
        adj_forward = adj_dense + torch.eye(G, device=adj_dense.device)
        mask_forward = (adj_forward == 0).float() * -1e9  # Mask for non-edges

        # Backward mask (Target → TF)
        adj_backward = adj_dense.T + torch.eye(G, device=adj_dense.device)
        mask_backward = (adj_backward == 0).float() * -1e9

        # Forward propagation: information flows from TF to targets
        out_forward, _ = self.attn_forward(
            x, x, x, attn_mask=mask_forward
        )

        # Backward propagation: information flows from targets to TFs
        out_backward, _ = self.attn_backward(
            x, x, x, attn_mask=mask_backward
        )

        # Combine directional information
        out = self.alpha * out_forward + self.beta * out_backward
        out = self.dropout(out)

        x = self.norm1(x + out)

        # Feed-forward transformation
        ffn_out = self.ffn(x)
        ffn_out = self.dropout(ffn_out)

        x = self.norm2(self.wolf * x + ffn_out)

        return x


# =====================================================
# 4️⃣ High-order Perception Enhancement
# Captures indirect regulatory relationships through 2-hop propagation
# =====================================================

class HighOrderPerceptionLayer(nn.Module):
    """
    High-order structure perception layer.
    Captures indirect regulatory effects through second-order adjacency.

    Args:
        dim: Dimension of node embeddings
    """

    def __init__(self, dim):
        super().__init__()

        self.linear = nn.Linear(dim, dim)

        # Controls high-order information strength
        self.gamma = nn.Parameter(torch.tensor(0.01), requires_grad=False)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, adj):
        """
        Forward pass for high-order structure perception.

        Args:
            x: Node embeddings (1, num_genes, dim)
            adj: Sparse adjacency matrix

        Returns:
            Updated node embeddings with high-order information
        """

        # x: (1, G, d)
        adj_dense = adj.to_dense()

        # ===== Second-order structure =====
        # A2 represents 2-hop connections (indirect regulations)
        A2 = torch.matmul(adj_dense, adj_dense)

        # Normalize by degree
        deg = A2.sum(dim=1, keepdim=True) + 1e-6
        A2_norm = A2 / deg

        A2_norm = A2_norm.unsqueeze(0)

        # High-order propagation
        h_ho = torch.matmul(A2_norm, x)

        h_ho = self.linear(h_ho)
        h_ho = F.gelu(h_ho)

        # Combine with original embeddings
        out = self.gamma * x + h_ho

        out = self.norm(out)

        return out

# =====================================================
# 5️⃣ Decoder - Predicts regulatory link probability
# =====================================================

class RegulationAwareDecoder(nn.Module):
    """
    Edge decoder for predicting regulatory link probability.
    Uses multiple feature concatenation strategies.

    Args:
        dim: Dimension of TF and target embeddings
    """
    def __init__(self, dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, h_i, h_j):
        """
        Decode regulatory link probability from TF and target embeddings.

        Args:
            h_i: TF embedding
            h_j: Target embedding

        Returns:
            Logit score for regulatory link
        """

        # Concatenate multiple feature combinations
        edge_feat = torch.cat([
            h_i,
            h_j,
            h_i - h_j,  # Difference features
            h_i * h_j   # Interaction features
        ], dim=1)

        return self.mlp(edge_feat)


# =====================================================
# 6️⃣ MechaGRN - Main model integrating all components
# =====================================================

class MechaGRN(nn.Module):
    """
    Mechanism-aware Gene Regulatory Network inference model.

    Architecture:
        1. Expression encoder with sparsity-aware gating
        2. Direction-aware graph attention layers
        3. High-order structure perception
        4. Regulatory link decoder

    Args:
        input_dim: Number of cells (expression matrix columns)
        hidden_dim: Hidden dimension size
        output_dim: Final embedding dimension
        num_heads: Number of attention heads
        device: Computing device (cuda/cpu)
        use_expression: Whether to use expression encoder
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        output_dim=16,
        num_heads=4,
        device="cuda",
        use_expression=True
    ):
        super().__init__()

        self.device = device
        self.use_expression = use_expression

        # Fallback projection if expression encoder is disabled
        if not use_expression:
            self.fallback_proj = nn.Linear(input_dim, hidden_dim)

        # Expression encoder
        self.expr_encoder = RegulationAwareExpressionTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )

        # Direction-aware graph layers
        self.graph_layer1 = DirectionAwareGraphLayer(hidden_dim, num_heads)
        self.graph_layer2 = DirectionAwareGraphLayer(hidden_dim, num_heads)

        # High-order perception layer
        self.high_order = HighOrderPerceptionLayer(hidden_dim)

        # TF and target specific projections
        self.tf_linear = nn.Linear(hidden_dim, output_dim)
        self.target_linear = nn.Linear(hidden_dim, output_dim)

        # Edge decoder
        self.decoder = RegulationAwareDecoder(output_dim)

    def forward(self, x, adj, train_sample):
        """
        Forward pass for GRN inference.

        Args:
            x: Gene expression matrix (num_genes, num_cells)
            adj: Sparse adjacency matrix
            train_sample: Training samples (TF_idx, Target_idx, Label)

        Returns:
            Predicted regulatory link scores
        """

        # Encode gene expression
        if self.use_expression:
            node_embed = self.expr_encoder(x)
        else:
            node_embed = self.fallback_proj(x)

        node_embed = node_embed.unsqueeze(0)  # Add batch dimension

        # Apply direction-aware graph attention
        node_embed = self.graph_layer1(node_embed, adj)
        node_embed = self.graph_layer2(node_embed, adj)

        # ===== High-order structure enhancement =====
        node_embed = self.high_order(node_embed, adj)

        node_embed = node_embed.squeeze(0)  # Remove batch dimension

        # Project TF and target embeddings
        tf_embed = F.elu(self.tf_linear(node_embed))
        target_embed = F.elu(self.target_linear(node_embed))

        # Extract TF and target indices from training samples
        tf_idx = train_sample[:, 0]
        target_idx = train_sample[:, 1]

        train_tf = tf_embed[tf_idx]
        train_target = target_embed[target_idx]

        # Decode regulatory link scores
        pred = self.decoder(train_tf, train_target)
        pred = torch.nan_to_num(pred)  # Handle NaN values

        return pred