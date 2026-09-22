import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def _dst_softmax_multihead(
    e_ij: torch.Tensor,
    dst_idx: torch.Tensor,
    num_dst: int,
    edge_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-9
) -> torch.Tensor:
    assert e_ij.dim() == 2, "e_ij should be [E, H]"
    E, H = e_ij.shape
    if E == 0:
        return e_ij.new_empty((0, H))
    logits = e_ij
    if edge_weight is not None:
        w = torch.clamp(edge_weight.to(e_ij.dtype), min=0.0)
        logits = logits + torch.log(w.view(-1, 1) + 1e-12)

    if hasattr(torch.Tensor, "scatter_reduce_"):
        max_per_dst = torch.full((num_dst, H), -1e15, device=logits.device, dtype=logits.dtype)
        idx2d = dst_idx.view(-1, 1).expand(-1, H)
        max_per_dst.scatter_reduce_(0, idx2d, logits, reduce="amax", include_self=True)
        logits = logits - max_per_dst[dst_idx]
    else:

        logits = logits - logits.max(dim=0, keepdim=True).values
    exp_e = torch.exp(logits)
    denom = torch.zeros((num_dst, H), device=logits.device, dtype=logits.dtype)
    denom.index_add_(0, dst_idx, exp_e)
    alpha = exp_e / (denom[dst_idx] + eps)
    return alpha

class BipartiteGATMultiHead(nn.Module):
    def __init__(
        self,
        in_src: int,
        in_dst: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.3,
        negative_slope: float = 0.2,
        concat: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if concat:
            assert out_dim % num_heads == 0, f"out_dim({out_dim}) must be divisible by num_heads({num_heads}) when concat=True"
            self.head_dim = out_dim // num_heads
            proj_out = num_heads * self.head_dim
        else:
            self.head_dim = out_dim
            proj_out = num_heads * self.head_dim

        self.lin_src = nn.Linear(in_src, proj_out, bias=False)
        self.lin_dst = nn.Linear(in_dst, proj_out, bias=True)

        self.att_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_src_idx: torch.Tensor,
        edge_dst_idx: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x_src.device
        assert edge_src_idx.dtype == torch.long and edge_dst_idx.dtype == torch.long
        assert edge_src_idx.device == device and edge_dst_idx.device == device

        Ns = x_src.size(0)
        Nd = x_dst.size(0)
        H = self.num_heads
        Dh = self.head_dim

        h_src = self.lin_src(x_src).view(Ns, H, Dh)
        h_dst = self.lin_dst(x_dst).view(Nd, H, Dh)

        hs_e = h_src[edge_src_idx]
        hd_e = h_dst[edge_dst_idx]

        e_ij = (hs_e * self.att_src.view(1, H, Dh)).sum(-1) + (hd_e * self.att_dst.view(1, H, Dh)).sum(-1)
        e_ij = self.leaky_relu(e_ij)

        alpha = _dst_softmax_multihead(e_ij, edge_dst_idx, Nd, edge_weight=edge_weight)
        alpha = self.dropout(alpha)

        msg = hs_e * alpha.unsqueeze(-1)
        agg = torch.zeros((Nd, H, Dh), device=device, dtype=h_src.dtype)
        agg.index_add_(0, edge_dst_idx, msg)

        out = h_dst + agg

        if self.concat:
            out = out.reshape(Nd, H * Dh)
        else:
            out = out.mean(dim=1)

        out = self.norm(out)
        out = F.elu(out)
        out = self.dropout(out)

        return out, alpha

class GatedMoEFusion(nn.Module):

    def __init__(self, dim: int, hidden: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if hidden is None:
            hidden = dim

        gate_in_dim = 4 * dim
        self.ln = nn.LayerNorm(gate_in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(
        self,
        z_know: torch.Tensor,
        z_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if z_know.shape != z_data.shape:
            raise ValueError("The two fusion views must have identical shapes")

        x = torch.cat(
            [z_know, z_data, (z_know - z_data).abs(), z_know * z_data], dim=-1
        )
        x = self.ln(x)
        g = torch.sigmoid(self.mlp(x))

        z = g * z_know + (1.0 - g) * z_data
        return z, g.squeeze(-1)

class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class DMPNNEncoder(nn.Module):
    def __init__(self, atom_dim, bond_dim, hidden_dim, out_dim, depth=3, dropout=0.2):
        super().__init__()
        if depth < 1:
            raise ValueError("DMPNN depth must be >= 1")
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.W_i = nn.Linear(atom_dim + bond_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(atom_dim + hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph):
        atoms, bonds = graph["atom_features"], graph["bond_features"]
        src, dst, rev = graph["edge_src"], graph["edge_dst"], graph["reverse_edge"]

        h0 = self.W_i(torch.cat([atoms[src], bonds], dim=-1))
        h = F.relu(h0)
        for _ in range(self.depth - 1):
            incoming = h.new_zeros((atoms.size(0), self.hidden_dim))
            incoming.index_add_(0, dst, h)

            message = incoming[src] - h[rev]
            h = self.dropout(F.relu(h0 + self.W_h(message)))
        incoming = h.new_zeros((atoms.size(0), self.hidden_dim))
        incoming.index_add_(0, dst, h)
        atom_hidden = self.dropout(F.relu(self.W_o(torch.cat([atoms, incoming], dim=-1))))
        mol_hidden = atom_hidden.new_zeros((graph["num_molecules"], atom_hidden.size(1)))
        mol_hidden.index_add_(0, graph["atom_to_molecule"], atom_hidden)
        return mol_hidden / graph["atom_counts"].to(atom_hidden.dtype).unsqueeze(-1)

class MultiViewModel(nn.Module):
    def __init__(
        self,
        atom_dim,
        bond_dim,
        in_target,
        num_pathways,
        in_cell_mut,
        in_cell_cnv,
        in_cell_rna,
        hidden,
        emb_dim,
        dropout,
        num_heads: int = 4,
        concat: bool = True,
        gate_hidden: Optional[int] = None,
        dmpnn_hidden: Optional[int] = None,
        dmpnn_depth: int = 3,
    ):
        super().__init__()
        self.drug_encoder = DMPNNEncoder(
            atom_dim, bond_dim, dmpnn_hidden or emb_dim, emb_dim,
            depth=dmpnn_depth, dropout=dropout,
        )
        self.pathway_embedding = nn.Embedding(num_pathways, emb_dim)
        gat_options = dict(
            out_dim=emb_dim, num_heads=num_heads,
            dropout=dropout, concat=concat,
        )

        self.gat_t2p = BipartiteGATMultiHead(in_target, emb_dim, **gat_options)
        self.gat_p2t = BipartiteGATMultiHead(emb_dim, in_target, **gat_options)
        self.gat_t2d = BipartiteGATMultiHead(emb_dim, emb_dim, **gat_options)
        self.gat_p2d = BipartiteGATMultiHead(emb_dim, emb_dim, **gat_options)
        self.fuse_drug = GatedMoEFusion(
            dim=emb_dim, hidden=gate_hidden, dropout=dropout,
        )
        self.cell_mut_encoder = nn.Sequential(
            nn.Linear(in_cell_mut, emb_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.cell_cnv_encoder = nn.Sequential(
            nn.Linear(in_cell_cnv, emb_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.cell_rna_encoder = nn.Sequential(
            nn.Linear(in_cell_rna, emb_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.gat_p2c = BipartiteGATMultiHead(emb_dim, emb_dim, **gat_options)
        self.head = MLPRegressor(
            in_dim=2 * emb_dim, hidden=hidden, dropout=dropout,
        )

    def forward(
        self,
        drug_graph, x_target, x_cell_mut, x_cell_cnv, x_cell_rna,
        edges_dt,
        edges_dp, w_dp,
        edges_pc, w_pc,
        edges_pt,
        pairs_drug_idx, pairs_cell_idx,
    ):

        x_drug = self.drug_encoder(drug_graph)

        z_pathway = self.pathway_embedding.weight

        z_pathway_star, attn_t2p = self.gat_t2p(
            x_target, z_pathway, *edges_pt["target2pathway"],
        )
        z_target_star, attn_p2t = self.gat_p2t(
            z_pathway, x_target, *edges_pt["pathway2target"],
        )

        z_drug_dt, attn_t2d = self.gat_t2d(
            z_target_star, x_drug, *edges_dt["target2drug"],
        )

        z_drug_dp, attn_p2d = self.gat_p2d(
            z_pathway_star, z_drug_dt, *edges_dp["pathway2drug"],
            edge_weight=w_dp["pathway2drug"],
        )

        z_drug, g_drug = self.fuse_drug(z_drug_dt, z_drug_dp)

        x_cell = (
            self.cell_mut_encoder(x_cell_mut)
            + self.cell_cnv_encoder(x_cell_cnv)
            + self.cell_rna_encoder(x_cell_rna)
        ) / 3.0

        z_cell, attn_p2c = self.gat_p2c(
            z_pathway_star, x_cell, *edges_pc["pathway2cell"],
            edge_weight=w_pc["pathway2cell"],
        )

        pred = self.head(torch.cat(
            [z_drug[pairs_drug_idx], z_cell[pairs_cell_idx]], dim=-1,
        ))
        return pred, {
            "attn_pt": {"target2pathway": attn_t2p, "pathway2target": attn_p2t},
            "attn_dt": {"target2drug": attn_t2d},
            "attn_dp": {"pathway2drug": attn_p2d},
            "attn_pc": {"pathway2cell": attn_p2c},
            "g_drug": g_drug,
        }
