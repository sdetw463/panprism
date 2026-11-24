"""
Edge-conditioned Siamese graph attention networks for breakpoint, splice-competition,
and disruption-magnitude scoring.

Provides three compact, parameter-efficient Siamese graph neural operators with exact
architectural invariants verified to machine zero (`0.0`):

1. `SiameseBreakpointGAT`:
   Edge-conditioned multi-head GAT over personal regulatory breakpoint graphs
   `(V_ref, V_ins, V_bkp, E_3D, E_order)` with factored edge attention (zero 5D tensors)
   and activity-gated insertion/breakpoint nodes (inert bubbles with `node_act == 0` contribute
   exact `0.0` attention weight and `0.0` message).
     - Exact antisymmetry:       `max |f(H1, H2) + f(H2, H1)| == 0.0`
     - Exact homozygous identity: `max |f(H, H)| == 0.0`

2. `SiameseSpliceCompetitionGAT`:
   Prior-anchored edge-conditioned Siamese GAT over topological splice-cluster graphs
   `(V_focal, V_comp, V_exon, V_bubble, E_5to3, E_shared_ss, E_rope, E_3D)` with monotonic
   focal-vs-competing weighting (`softplus(w_prior + dw)`) and learned competitive attention.
     - Exact antisymmetry:       `max |f(H1, H2) + f(H2, H1)| == 0.0`
     - Exact homozygous identity: `max |f(H, H)| == 0.0`

3. `SiameseMagnitudeGAT`:
   Swap-invariant Siamese disruption-magnitude network `g(H1, H2) = mean(||Phi(H1) - Phi(H2)||^2)`
   for unsigned rare/singleton SV outlier and clinical disruption scoring.
     - Exact swap invariance:    `max |g(H1, H2) - g(H2, H1)| == 0.0`
     - Exact homozygous identity: `max |g(H, H)| == 0.0`

PyTorch is an optional dependency: importing `panprism` or `panprism.siamese_gat`
succeeds cleanly in lightweight NumPy-only environments (`HAS_TORCH == False`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Sequence, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
    _ModuleBase: Any = nn.Module
    _NO_GRAD = torch.no_grad()
except ImportError:  # pragma: no cover - lightweight CPU env without torch
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    HAS_TORCH = False
    _ModuleBase = object

    def _NO_GRAD(fn: Any) -> Any:  # type: ignore
        return fn


def _require_torch() -> None:
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for panprism.siamese_gat models. "
            "Install with `pip install 'panprism[gpu]'` or `pip install torch`."
        )


@dataclass
class PairedGraphBatch:
    """Paired `(H1, H2)` haplotype breakpoint-graph batch for Siamese training and inference."""

    # Haplotype 1 (e.g., H1 or ALT)
    node_x1: Any       # (B, M, D_v) float32
    node_type1: Any    # (B, M) int64
    node_act1: Any     # (B, M, 1) float32 in [0, 1]
    node_mask1: Any    # (B, M) bool
    edge_x1: Any       # (B, M, M, D_e) float32
    edge_mask1: Any    # (B, M, M) bool
    readout_w1: Any    # (B, M, R) float32

    # Haplotype 2 (e.g., H2 or REF)
    node_x2: Any
    node_type2: Any
    node_act2: Any
    node_mask2: Any
    edge_x2: Any
    edge_mask2: Any
    readout_w2: Any

    def __post_init__(self) -> None:
        _require_torch()

    def to(self, device: Any) -> "PairedGraphBatch":
        return PairedGraphBatch(
            node_x1=self.node_x1.to(device),
            node_type1=self.node_type1.to(device),
            node_act1=self.node_act1.to(device),
            node_mask1=self.node_mask1.to(device),
            edge_x1=self.edge_x1.to(device),
            edge_mask1=self.edge_mask1.to(device),
            readout_w1=self.readout_w1.to(device),
            node_x2=self.node_x2.to(device),
            node_type2=self.node_type2.to(device),
            node_act2=self.node_act2.to(device),
            node_mask2=self.node_mask2.to(device),
            edge_x2=self.edge_x2.to(device),
            edge_mask2=self.edge_mask2.to(device),
            readout_w2=self.readout_w2.to(device),
        )

    def slice_rows(self, idx: Union[Any, np.ndarray]) -> "PairedGraphBatch":
        if isinstance(idx, np.ndarray):
            if idx.dtype == bool:
                idx = np.nonzero(idx)[0]
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=self.node_x1.device)
        else:
            idx_t = idx
        return PairedGraphBatch(
            node_x1=self.node_x1[idx_t],
            node_type1=self.node_type1[idx_t],
            node_act1=self.node_act1[idx_t],
            node_mask1=self.node_mask1[idx_t],
            edge_x1=self.edge_x1[idx_t],
            edge_mask1=self.edge_mask1[idx_t],
            readout_w1=self.readout_w1[idx_t],
            node_x2=self.node_x2[idx_t],
            node_type2=self.node_type2[idx_t],
            node_act2=self.node_act2[idx_t],
            node_mask2=self.node_mask2[idx_t],
            edge_x2=self.edge_x2[idx_t],
            edge_mask2=self.edge_mask2[idx_t],
            readout_w2=self.readout_w2[idx_t],
        )

    def swap_haplotypes(self) -> "PairedGraphBatch":
        """Return `(H2, H1)` batch for exact antisymmetry / swap-invariance verification."""
        return PairedGraphBatch(
            node_x1=self.node_x2,
            node_type1=self.node_type2,
            node_act1=self.node_act2,
            node_mask1=self.node_mask2,
            edge_x1=self.edge_x2,
            edge_mask1=self.edge_mask2,
            readout_w1=self.readout_w2,
            node_x2=self.node_x1,
            node_type2=self.node_type1,
            node_act2=self.node_act1,
            node_mask2=self.node_mask1,
            edge_x2=self.edge_x1,
            edge_mask2=self.edge_mask1,
            readout_w2=self.readout_w1,
        )

    def identical_pair(self) -> "PairedGraphBatch":
        """Return homozygous `(H1, H1)` batch for exact zero-SV identity verification."""
        return PairedGraphBatch(
            node_x1=self.node_x1,
            node_type1=self.node_type1,
            node_act1=self.node_act1,
            node_mask1=self.node_mask1,
            edge_x1=self.edge_x1,
            edge_mask1=self.edge_mask1,
            readout_w1=self.readout_w1,
            node_x2=self.node_x1,
            node_type2=self.node_type1,
            node_act2=self.node_act1,
            node_mask2=self.node_mask1,
            edge_x2=self.edge_x1,
            edge_mask2=self.edge_mask1,
            readout_w2=self.readout_w1,
        )


class EdgeConditionedGATBlock(_ModuleBase):
    """Single Edge-Conditioned Multi-Head Graph Attention layer with activity-gated message passing."""

    def __init__(self, hidden_dim: int, edge_dim: int, num_heads: int = 4, dropout: float = 0.2):
        _require_torch()
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.e_k_proj = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.e_v_proj = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.e_bias = nn.Linear(edge_dim, num_heads, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h: Any,          # (B, M, D)
        e_emb: Any,      # (B, M, M, D_e)
        edge_mask: Any,  # (B, M, M) bool
        node_mask: Any,  # (B, M) bool
        src_gate: Any,   # (B, M, 1) float in [0, 1]
    ) -> Any:
        B, M, D = h.shape
        H, dk = self.num_heads, self.head_dim
        hn = self.norm1(h)

        q = self.q_proj(hn).view(B, M, H, dk).permute(0, 2, 1, 3)  # (B, H, M, dk)
        k = self.k_proj(hn).view(B, M, H, dk).permute(0, 2, 1, 3)  # (B, H, M, dk)
        v = self.v_proj(hn).view(B, M, H, dk).permute(0, 2, 1, 3)  # (B, H, M, dk)
        eb = self.e_bias(e_emb).permute(0, 3, 1, 2)               # (B, H, M, M)

        # Factored edge-conditioned attention logits (zero 5D allocations):
        # q_u^T (e_{uv} @ W_k^T) == (q_u @ W_k) @ e_{uv}^T
        w_ek = self.e_k_proj.weight.view(H, dk, -1)
        w_ev = self.e_v_proj.weight.view(H, dk, -1)
        q_e = torch.matmul(q, w_ek)
        qk_node = torch.matmul(q, k.transpose(-2, -1))
        qk_edge = torch.matmul(q_e.permute(0, 2, 1, 3), e_emb.transpose(-2, -1)).permute(0, 2, 1, 3)
        logits = (qk_node + qk_edge) / math.sqrt(dk) + eb

        # Guarantee valid self-loop for canonical nodes and zero attention for inert bubbles
        eye = torch.eye(M, dtype=torch.bool, device=h.device).unsqueeze(0)
        valid_src = node_mask & (src_gate.squeeze(-1) > 1e-6)
        valid_edges = (edge_mask | eye) & node_mask.unsqueeze(2) & valid_src.unsqueeze(1)
        valid_edges = valid_edges | (eye & node_mask.unsqueeze(2))

        logits = logits.masked_fill(~valid_edges.unsqueeze(1), -1e9)
        alpha = torch.softmax(logits, dim=-1) * valid_edges.unsqueeze(1).float()
        alpha = self.drop(alpha)
        alpha_g = alpha * src_gate.view(B, 1, 1, M)
        pooled_e = torch.matmul(alpha_g.permute(0, 2, 1, 3), e_emb)
        agg_edge = torch.matmul(pooled_e.permute(0, 2, 1, 3), w_ev.transpose(-2, -1))
        agg = torch.matmul(alpha_g, v) + agg_edge
        agg = agg.permute(0, 2, 1, 3).contiguous().view(B, M, D)

        h = h + self.drop(self.out_proj(agg))
        h = h + self.ffn(self.norm2(h))
        return h * node_mask.unsqueeze(-1).float()


class SiameseBreakpointGAT(_ModuleBase):
    """
    Edge-conditioned Siamese breakpoint graph attention network.

    Computes `f(H1, H2) = Phi(G(H1)) - Phi(G(H2))`, guaranteeing exact machine-zero
    antisymmetry `f(H1, H2) = -f(H2, H1)` and homozygous identity `f(H, H) = 0`.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        readout_dim: int = 2,
        n_node_types: int = 6,
        hidden_dim: int = 32,
        edge_hidden_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        _require_torch()
        super().__init__()
        self.node_in = nn.Linear(node_dim, hidden_dim, bias=False)
        self.type_emb = nn.Embedding(n_node_types, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_in = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim, bias=False),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [
                EdgeConditionedGATBlock(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.readout_norm = nn.LayerNorm(hidden_dim)
        self.raw_skip = nn.Linear(node_dim * readout_dim, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * readout_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_single(
        self,
        node_x: Any,
        node_type: Any,
        node_act: Any,
        node_mask: Any,
        edge_x: Any,
        edge_mask: Any,
        readout_w: Any,
    ) -> Any:
        nt_safe = node_type.clamp(0, self.type_emb.num_embeddings - 1)
        h = self.node_in(node_x) + self.type_emb(nt_safe)
        h = F.gelu(self.node_norm(h)) * node_act * node_mask.unsqueeze(-1).float()
        e_emb = self.edge_in(edge_x)

        for layer in self.layers:
            h = layer(h, e_emb, edge_mask, node_mask, node_act)

        hn = self.readout_norm(h)
        pooled = torch.einsum("bmr,bmd->brd", readout_w, hn).reshape(hn.shape[0], -1)
        raw_r = torch.einsum("bmr,bmd->brd", readout_w, node_x * node_act).reshape(hn.shape[0], -1)
        return self.head(pooled).squeeze(-1) + self.raw_skip(raw_r).squeeze(-1)

    def forward_pair(self, batch: PairedGraphBatch) -> Any:
        phi1 = self.encode_single(
            batch.node_x1,
            batch.node_type1,
            batch.node_act1,
            batch.node_mask1,
            batch.edge_x1,
            batch.edge_mask1,
            batch.readout_w1,
        )
        phi2 = self.encode_single(
            batch.node_x2,
            batch.node_type2,
            batch.node_act2,
            batch.node_mask2,
            batch.edge_x2,
            batch.edge_mask2,
            batch.readout_w2,
        )
        return phi1 - phi2

    def forward(self, batch: PairedGraphBatch) -> Any:
        return self.forward_pair(batch)

    @_NO_GRAD
    def predict_antisymmetric(self, batch: PairedGraphBatch) -> np.ndarray:
        self.eval()
        phi1 = self.encode_single(
            batch.node_x1,
            batch.node_type1,
            batch.node_act1,
            batch.node_mask1,
            batch.edge_x1,
            batch.edge_mask1,
            batch.readout_w1,
        ).double()
        phi2 = self.encode_single(
            batch.node_x2,
            batch.node_type2,
            batch.node_act2,
            batch.node_mask2,
            batch.edge_x2,
            batch.edge_mask2,
            batch.readout_w2,
        ).double()
        return (phi1 - phi2).detach().cpu().numpy()


class SiameseSpliceCompetitionGAT(_ModuleBase):
    """
    Edge-conditioned Siamese splice-competition graph attention network.

    Encodes a personal splice-cluster graph `(V_focal, V_comp, V_exon, V_bubble, E_5to3,
    E_shared_ss, E_rope, E_3D)` using competitive focal-vs-competing attention and monotonic
    biophysical priors (`softplus(w_prior + dw)`).

    Guarantees exact machine-zero antisymmetry `f(H1, H2) = -f(H2, H1)` and exact
    homozygous identity `f(H, H) = 0`.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        w_prior_init: Optional[Union[Any, Sequence[float]]] = None,
        n_node_types: int = 6,
        hidden_dim: int = 16,
        num_heads: int = 4,
        dropout: float = 0.35,
        res_beta: float = 0.15,
    ):
        _require_torch()
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.dk = hidden_dim // num_heads
        self.res_beta = float(res_beta)

        if w_prior_init is None:
            prior_t = torch.zeros(node_dim, dtype=torch.float32)
        else:
            prior_t = torch.as_tensor(w_prior_init, dtype=torch.float32).clone().detach()
        if prior_t.shape != (node_dim,):
            raise ValueError(f"w_prior_init must have shape ({node_dim},), got {tuple(prior_t.shape)}")

        self.register_buffer("w_prior", prior_t)
        self.dw = nn.Parameter(torch.zeros(node_dim))
        self.type_emb = nn.Embedding(n_node_types, hidden_dim)
        self.node_proj = nn.Linear(node_dim, hidden_dim, bias=False)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.e_bias = nn.Sequential(
            nn.Linear(edge_dim, 8),
            nn.GELU(),
            nn.Linear(8, num_heads, bias=False),
        )
        self.res_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def encode_single(
        self,
        nx: Any,
        nt: Any,
        na: Any,
        nm: Any,
        ex: Any,
        em: Any,
        rw: Any,
    ) -> Any:
        B, M, _ = nx.shape
        H, dk = self.num_heads, self.dk
        nt_safe = nt.clamp(0, self.type_emb.num_embeddings - 1)
        h0 = F.gelu(self.node_proj(nx) + self.type_emb(nt_safe)) * na * nm.unsqueeze(-1).float()
        q = self.q_proj(h0).view(B, M, H, dk).permute(0, 2, 1, 3)
        k = self.k_proj(h0).view(B, M, H, dk).permute(0, 2, 1, 3)
        v = self.v_proj(h0).view(B, M, H, dk).permute(0, 2, 1, 3)
        eb = self.e_bias(ex).permute(0, 3, 1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk) + eb

        valid_src = nm & (na.squeeze(-1) > 1e-6)
        eye = torch.eye(M, dtype=torch.bool, device=nx.device).unsqueeze(0)
        valid = (em & nm.unsqueeze(2) & valid_src.unsqueeze(1)) | (eye & nm.unsqueeze(2))
        logits = logits.masked_fill(~valid.unsqueeze(1), -1e9)
        alpha = torch.softmax(logits, dim=-1) * valid.unsqueeze(1).float()
        att_0 = alpha[:, :, 0, :].mean(dim=1)

        rw_comp = rw[:, :, 1]
        comp_m = (rw_comp > 0).float()
        w_att = att_0 * comp_m
        w_att = w_att / (w_att.sum(dim=1, keepdim=True) + 1e-6)
        has_c = (comp_m.sum(dim=1, keepdim=True) > 0).float()
        w_comp = 0.65 * rw_comp + 0.35 * w_att * has_c

        x_foc = nx[:, 0, :]
        x_comp = torch.bmm(w_comp.unsqueeze(1), nx).squeeze(1)
        w_eff = F.softplus(self.w_prior + self.dw)
        score = ((x_foc - x_comp) * w_eff).sum(dim=-1)

        if self.res_beta > 0.0:
            h_msg = torch.matmul(alpha, v).permute(0, 2, 1, 3).contiguous().view(B, M, -1)
            h_pool = h_msg[:, 0, :] - torch.bmm(w_comp.unsqueeze(1), h_msg).squeeze(1)
            score = score + self.res_beta * self.res_head(h_pool).squeeze(-1)
        return score

    def forward_pair(self, batch: PairedGraphBatch) -> Any:
        return self.encode_single(
            batch.node_x1,
            batch.node_type1,
            batch.node_act1,
            batch.node_mask1,
            batch.edge_x1,
            batch.edge_mask1,
            batch.readout_w1,
        ) - self.encode_single(
            batch.node_x2,
            batch.node_type2,
            batch.node_act2,
            batch.node_mask2,
            batch.edge_x2,
            batch.edge_mask2,
            batch.readout_w2,
        )

    def forward(self, batch: PairedGraphBatch) -> Any:
        return self.forward_pair(batch)

    @_NO_GRAD
    def predict_antisymmetric(self, batch: PairedGraphBatch) -> np.ndarray:
        self.eval()
        p1 = self.encode_single(
            batch.node_x1,
            batch.node_type1,
            batch.node_act1,
            batch.node_mask1,
            batch.edge_x1,
            batch.edge_mask1,
            batch.readout_w1,
        ).double()
        p2 = self.encode_single(
            batch.node_x2,
            batch.node_type2,
            batch.node_act2,
            batch.node_mask2,
            batch.edge_x2,
            batch.edge_mask2,
            batch.readout_w2,
        ).double()
        return (p1 - p2).detach().cpu().numpy()


class SiameseMagnitudeGAT(_ModuleBase):
    """
    Swap-Invariant Siamese Breakpoint-Graph Outlier Magnitude Network:

        g(H1, H2) = mean(||Phi_theta(G(H1)) - Phi_theta(G(H2))||^2)

    Guarantees exact machine-zero swap invariance `g(H1, H2) == g(H2, H1)` and exact
    homozygous zero-SV identity `g(H, H) == 0.0`.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        readout_dim: int = 2,
        num_node_types: int = 6,
        hidden_dim: int = 16,
        edge_hidden_dim: int = 8,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.25,
    ):
        _require_torch()
        super().__init__()
        self.type_emb = nn.Embedding(num_node_types, hidden_dim)
        self.node_in = nn.Linear(node_dim, hidden_dim, bias=False)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_in = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim, bias=False),
            nn.GELU(),
        )
        self.act_scale = nn.Parameter(torch.ones(1))
        self.layers = nn.ModuleList(
            [
                EdgeConditionedGATBlock(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.vec_head = nn.Linear(hidden_dim * (readout_dim + 1), hidden_dim, bias=False)
        self.raw_scale = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(-1.0))

    def encode_vec(
        self,
        node_x: Any,
        node_type: Any,
        node_act: Any,
        node_mask: Any,
        edge_x: Any,
        edge_mask: Any,
        readout_w: Any,
    ) -> Any:
        gate = node_act.clamp(min=0.0, max=2.0) * F.softplus(self.act_scale)
        h = self.node_in(node_x) + self.type_emb(node_type.clamp(0, self.type_emb.num_embeddings - 1))
        h = F.gelu(self.node_norm(h)) * gate * node_mask.unsqueeze(-1).float()
        e_emb = self.edge_in(edge_x)
        for blk in self.layers:
            h = blk(h, e_emb, edge_mask, node_mask, gate)
        h = self.norm_out(h) * node_mask.unsqueeze(-1).float()
        w_norm = readout_w / readout_w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        r_pools = torch.bmm(w_norm.transpose(1, 2), h).reshape(h.shape[0], -1)
        g_pool = (h * gate).sum(dim=1) / node_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        return self.vec_head(torch.cat([r_pools, g_pool], dim=-1))

    def forward_dist2(self, batch: PairedGraphBatch) -> Any:
        v1 = self.encode_vec(
            batch.node_x1,
            batch.node_type1,
            batch.node_act1,
            batch.node_mask1,
            batch.edge_x1,
            batch.edge_mask1,
            batch.readout_w1,
        )
        v2 = self.encode_vec(
            batch.node_x2,
            batch.node_type2,
            batch.node_act2,
            batch.node_mask2,
            batch.edge_x2,
            batch.edge_mask2,
            batch.readout_w2,
        )
        return ((v1 - v2) ** 2).mean(dim=-1)

    def forward_logit(self, batch: PairedGraphBatch) -> Any:
        return self.bias + F.softplus(self.raw_scale) * self.forward_dist2(batch)

    def forward(self, batch: PairedGraphBatch) -> Any:
        return self.forward_dist2(batch)

    @_NO_GRAD
    def predict_mag(self, batch: PairedGraphBatch) -> np.ndarray:
        self.eval()
        v1 = self.encode_vec(
            batch.node_x1,
            batch.node_type1,
            batch.node_act1,
            batch.node_mask1,
            batch.edge_x1,
            batch.edge_mask1,
            batch.readout_w1,
        ).double()
        v2 = self.encode_vec(
            batch.node_x2,
            batch.node_type2,
            batch.node_act2,
            batch.node_mask2,
            batch.edge_x2,
            batch.edge_mask2,
            batch.readout_w2,
        ).double()
        return ((v1 - v2) ** 2).mean(dim=-1).detach().cpu().numpy()


class Matched1DSiameseMLP(_ModuleBase):
    """Capacity-matched nonlinear 1D Siamese MLP baseline `f(x1, x2) = psi(x1) - psi(x2)`."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        _require_torch()
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
            nn.LayerNorm(max(1, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, hidden_dim // 2), 1),
        )

    def forward_pair(self, x1: Any, x2: Any) -> Any:
        return self.net(x1).squeeze(-1) - self.net(x2).squeeze(-1)

    def forward(self, x1: Any, x2: Any) -> Any:
        return self.forward_pair(x1, x2)

    @_NO_GRAD
    def predict_antisymmetric(self, x1: Any, x2: Any) -> np.ndarray:
        self.eval()
        p1 = self.net(x1).squeeze(-1).double()
        p2 = self.net(x2).squeeze(-1).double()
        return (p1 - p2).detach().cpu().numpy()


class AnchoredMatched1DSiamese(_ModuleBase):
    """Biophysical prior-anchored capacity-matched 1D Siamese baseline."""

    def __init__(
        self,
        in_dim: int,
        w_prior_init: Optional[Union[Any, Sequence[float]]] = None,
        hidden_dim: int = 16,
        dropout: float = 0.35,
        res_beta: float = 0.15,
    ):
        _require_torch()
        super().__init__()
        if w_prior_init is None:
            prior_t = torch.zeros(in_dim, dtype=torch.float32)
        else:
            prior_t = torch.as_tensor(w_prior_init, dtype=torch.float32).clone().detach()
        self.register_buffer("w_prior", prior_t)
        self.dw = nn.Parameter(torch.zeros(in_dim))
        self.res_beta = float(res_beta)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def encode_single(self, x: Any) -> Any:
        return (x * F.softplus(self.w_prior + self.dw)).sum(dim=-1) + self.res_beta * self.mlp(x).squeeze(-1)

    def forward_pair(self, x1: Any, x2: Any) -> Any:
        return self.encode_single(x1) - self.encode_single(x2)

    def forward(self, x1: Any, x2: Any) -> Any:
        return self.forward_pair(x1, x2)

    @_NO_GRAD
    def predict_antisymmetric(self, x1: Any, x2: Any) -> np.ndarray:
        self.eval()
        return (self.encode_single(x1).double() - self.encode_single(x2).double()).detach().cpu().numpy()


def count_parameters(model: Any) -> int:
    """Count trainable parameters in a PyTorch module."""
    _require_torch()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_siamese_invariants(model: Any, batch: PairedGraphBatch) -> Dict[str, float]:
    """
    Numerically verify exact architectural invariants on `batch`:
      - For `SiameseMagnitudeGAT`: `max |g(H1, H2) - g(H2, H1)|` and `max |g(H, H)|`.
      - For `SiameseBreakpointGAT` & `SiameseSpliceCompetitionGAT`:
        `max |f(H1, H2) + f(H2, H1)|` and `max |f(H, H)|`.
    """
    _require_torch()
    swapped = batch.swap_haplotypes()
    ident = batch.identical_pair()

    if isinstance(model, SiameseMagnitudeGAT):
        g12 = model.predict_mag(batch)
        g21 = model.predict_mag(swapped)
        g11 = model.predict_mag(ident)
        return {
            "max_swap_err": float(np.max(np.abs(g12 - g21))),
            "max_zero_sv_err": float(np.max(np.abs(g11))),
        }

    f12 = model.predict_antisymmetric(batch)
    f21 = model.predict_antisymmetric(swapped)
    f11 = model.predict_antisymmetric(ident)
    return {
        "max_antisym_err": float(np.max(np.abs(f12 + f21))),
        "max_zero_sv_err": float(np.max(np.abs(f11))),
    }
