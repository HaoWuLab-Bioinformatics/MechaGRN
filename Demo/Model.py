import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


# =====================================================
# 1️⃣ Focal Loss
# =====================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        prob = torch.sigmoid(logits)
        pt = prob * targets + (1 - prob) * (1 - targets)

        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


# =====================================================
# 2️⃣ Expression Encoder
# =====================================================

class RegulationAwareExpressionTransformer(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, num_heads=4, dropout=0.1):
        super().__init__()

        self.expr_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        self.stat_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        self.sparsity_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )

        # ===== Self-Attention Layer =====
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.attn_norm = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):

        # expression projection
        expr_embed = self.expr_proj(x)

        # statistics
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True)
        dropout_rate = torch.mean((x == 0).float(), dim=1, keepdim=True)

        stats = torch.cat([mean, var, dropout_rate], dim=1)
        stat_embed = self.stat_proj(stats)

        # sparsity-aware gating
        gate = self.sparsity_gate(x)
        expr_embed = expr_embed * gate

        # feature fusion
        fusion_input = torch.cat([expr_embed, stat_embed], dim=1)
        h = self.fusion(fusion_input)

        # ===== Self-attention refinement =====
        h = h.unsqueeze(0)  # (1, N, hidden_dim)

        attn_out, _ = self.attn(h, h, h)
        h = self.attn_norm(h + attn_out)

        # ffn_out = self.ffn(h)
        # h = self.ffn_norm(h + ffn_out)

        h = h.squeeze(0)

        return h


# =====================================================
# 3️⃣ Direction-Aware Graph Transformer Layer
# =====================================================

class DirectionAwareGraphLayer(nn.Module):

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.attn_forward = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.attn_backward = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.alpha = nn.Parameter(torch.tensor(0.9), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(0.1), requires_grad= False)
        self.wolf = nn.Parameter(torch.tensor(0.01), requires_grad=False)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):

        B, G, d = x.size()
        adj_dense = adj.to_dense()

        # Forward mask (TF → Target)
        adj_forward = adj_dense + torch.eye(G, device=adj_dense.device)
        mask_forward = (adj_forward == 0).float() * -1e9

        # Backward mask (Target → TF)
        adj_backward = adj_dense.T + torch.eye(G, device=adj_dense.device)
        mask_backward = (adj_backward == 0).float() * -1e9

        # Forward propagation
        out_forward, _ = self.attn_forward(
            x, x, x, attn_mask=mask_forward
        )

        # Backward propagation
        out_backward, _ = self.attn_backward(
            x, x, x, attn_mask=mask_backward
        )

        out = self.alpha * out_forward + self.beta * out_backward
        out = self.dropout(out)

        x = self.norm1(x + out)

        ffn_out = self.ffn(x)
        ffn_out = self.dropout(ffn_out)

        x = self.norm2(self.wolf * x + ffn_out)

        return x


# =====================================================
# 4️⃣ High-order Perception Enhancement
# =====================================================

class HighOrderPerceptionLayer(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.linear = nn.Linear(dim, dim)

        # 控制高阶信息强度
        self.gamma = nn.Parameter(torch.tensor(0.01), requires_grad=False)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, adj):

        # x: (1, G, d)
        adj_dense = adj.to_dense()

        # ===== 二阶结构 =====
        A2 = torch.matmul(adj_dense, adj_dense)

        # normalization
        deg = A2.sum(dim=1, keepdim=True) + 1e-6
        A2_norm = A2 / deg

        A2_norm = A2_norm.unsqueeze(0)

        # 高阶传播
        h_ho = torch.matmul(A2_norm, x)

        h_ho = self.linear(h_ho)
        h_ho = F.gelu(h_ho)

        # 融合
        out = self.gamma * x + h_ho

        out = self.norm(out)

        return out

# =====================================================
# 4️⃣ Decoder
# =====================================================

class RegulationAwareDecoder(nn.Module):
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

        edge_feat = torch.cat([
            h_i,
            h_j,
            h_i - h_j,
            h_i * h_j
        ], dim=1)

        return self.mlp(edge_feat)


# =====================================================
# 5️⃣ GRANet (Direction-Aware Version)
# =====================================================

class MechaGRN(nn.Module):

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

        if not use_expression:
            self.fallback_proj = nn.Linear(input_dim, hidden_dim)

        self.expr_encoder = RegulationAwareExpressionTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )

        self.graph_layer1 = DirectionAwareGraphLayer(hidden_dim, num_heads)
        self.graph_layer2 = DirectionAwareGraphLayer(hidden_dim, num_heads)
        self.high_order = HighOrderPerceptionLayer(hidden_dim)

        self.tf_linear = nn.Linear(hidden_dim, output_dim)
        self.target_linear = nn.Linear(hidden_dim, output_dim)

        self.decoder = RegulationAwareDecoder(output_dim)

    def forward(self, x, adj, train_sample):

        if self.use_expression:
            node_embed = self.expr_encoder(x)
        else:
            node_embed = self.fallback_proj(x)

        node_embed = node_embed.unsqueeze(0)

        node_embed = self.graph_layer1(node_embed, adj)
        node_embed = self.graph_layer2(node_embed, adj)

        # ===== 高阶结构增强 =====
        node_embed = self.high_order(node_embed, adj)

        node_embed = node_embed.squeeze(0)

        tf_embed = F.elu(self.tf_linear(node_embed))
        target_embed = F.elu(self.target_linear(node_embed))

        tf_idx = train_sample[:, 0]
        target_idx = train_sample[:, 1]

        train_tf = tf_embed[tf_idx]
        train_target = target_embed[target_idx]

        pred = self.decoder(train_tf, train_target)
        pred = torch.nan_to_num(pred)

        return pred