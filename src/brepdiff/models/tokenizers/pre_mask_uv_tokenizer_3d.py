import math
from typing import Optional
import torch
import torch.nn as nn
from brepdiff.models.tokenizers.base import Tokens, Tokenizer
from brepdiff.primitives.uvgrid import UvGrid
import torch.nn.functional as F


class DenseGraphConvLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = valid_mask.float()
        adj = adj.float()

        # Mask invalid nodes and keep self-loop for valid nodes.
        adj = adj * valid.unsqueeze(1) * valid.unsqueeze(2)
        adj = adj + torch.diag_embed(valid)

        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        adj_norm = adj / deg

        h = torch.matmul(adj_norm, x)
        h = self.linear(h)
        h = F.gelu(h)
        return h


class DenseGraphAttnLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = valid_mask.bool()
        adj = adj.float()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])

        conn = adj > 0
        conn = conn | torch.eye(adj.shape[-1], device=adj.device, dtype=torch.bool)[None]
        conn = conn & valid.unsqueeze(1) & valid.unsqueeze(2)

        scores = scores.masked_fill(~conn, -1e4)
        attn = torch.softmax(scores, dim=-1)
        attn = attn * conn.float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        h = torch.matmul(attn, v)
        h = self.out_proj(h)
        h = F.gelu(h)
        return h


class FaceAdjGraphConditionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        z_dim: int,
        encoder_type: str = "gcn",
        num_layers: int = 2,
        hidden_dim: int = 256,
        pool: str = "mean",
    ):
        super().__init__()
        self.pool = pool

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        if encoder_type == "gat":
            self.layers = nn.ModuleList(
                [DenseGraphAttnLayer(hidden_dim) for _ in range(num_layers)]
            )
        elif encoder_type == "gcn":
            self.layers = nn.ModuleList(
                [DenseGraphConvLayer(hidden_dim) for _ in range(num_layers)]
            )
        else:
            raise ValueError(f"Unknown graph_encoder_type: {encoder_type}")

        self.out_proj = nn.Linear(hidden_dim, z_dim)

    def forward(
        self,
        face_adj: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Use topology only: derive per-node scalar from adjacency (node degree)
        # and tile it to match the expected input projection size.
        face_adj = face_adj.float()
        deg = face_adj.sum(dim=-1, keepdim=True)
        topo_feats = deg.repeat(1, 1, self.input_proj.in_features)
        h = self.input_proj(topo_feats)

        # valid_mask: (B, N) bool — True for real nodes, False for padding.
        # If not provided, treat all nodes as valid.
        if valid_mask is None:
            valid_mask = torch.ones(
                face_adj.shape[0],
                face_adj.shape[1],
                dtype=torch.bool,
                device=face_adj.device,
            )

        for layer in self.layers:
            h = layer(h, face_adj, valid_mask)

        # Masked pooling: ignore padding nodes so they don't dilute the signal.
        valid_float = valid_mask.float().unsqueeze(-1)  # (B, N, 1)
        if self.pool == "max":
            h_masked = h.masked_fill(~valid_mask.unsqueeze(-1), float("-inf"))
            pooled = h_masked.max(dim=1).values
        else:
            pooled = (h * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp(min=1.0)

        z = self.out_proj(pooled).unsqueeze(1)
        return z


class PreMaskUvTokenizer3D(Tokenizer):
    name = "pre_mask_uv_tokenizer_3d"
    """
    Append an empty mask to the token
    """

    def __init__(self, config, acc_loger):
        super().__init__(config, acc_loger)
        self.use_graph_conditioning = bool(
            getattr(self.config, "graph_conditioning", True)
        )

        self.graph_encoder = None
        if self.use_graph_conditioning:
            graph_input_dim = 3 + 1 + 1 + int(self.config.n_face_types > 0)
            self.graph_encoder = FaceAdjGraphConditionEncoder(
                input_dim=graph_input_dim,
                z_dim=self.config.z_dim,
                encoder_type=str(getattr(self.config, "graph_encoder_type", "gcn")).lower(),
                num_layers=int(getattr(self.config, "graph_num_layers", 2)),
                hidden_dim=int(getattr(self.config, "graph_hidden_dim", self.config.z_dim)),
                pool=str(getattr(self.config, "graph_pool", "mean")).lower(),
            )

    def forward(self, labels: torch.Tensor, uvgrids: UvGrid) -> Tokens:
        """
        :param labels: Tensor of B x max_n_prims
            one hot encoding of slicers
        :param uvgrids: UVGrid of coordinates having B x max_n_prims x n_grid x n_grid x data_dim
            could also have normals
        :return:
        """
        batch_size, max_n_prims, n_grid, _, _ = uvgrids.coord.shape
        # note that 1 indicates empty and -1 indicates not empty
        uvgrid_empty_mask = (2 * uvgrids.empty_mask - 1) > 0  # B x n_prims x 1

        # tensorize
        uvgrid_tensor = [uvgrids.coord]
        # B x n_prims x n_grid x n_grid x 1
        grid_mask = 2 * uvgrids.grid_mask.unsqueeze(-1).float() - 1
        uvgrid_tensor.append(grid_mask)

        # Append normalized integer face type index per grid point.
        if uvgrids.prim_type is not None:
            prim_type = uvgrids.prim_type.float()
            if prim_type.ndim == 2:
                prim_type = prim_type.unsqueeze(-1)

            if self.config.n_face_types > 1:
                prim_type = 2 * prim_type / (self.config.n_face_types - 1) - 1
            else:
                prim_type = torch.zeros_like(prim_type)

            if uvgrids.empty_mask is not None:
                prim_type = prim_type.clone()
                prim_type[uvgrids.empty_mask] = 0.0

            type_grid = prim_type.unsqueeze(2).unsqueeze(3).expand(
                -1, -1, n_grid, n_grid, -1
            )
            uvgrid_tensor.append(type_grid)

        uvgrid_tensor = torch.cat(uvgrid_tensor, dim=-1)
        uvgrid_tensor = uvgrid_tensor.view(batch_size, max_n_prims, -1)

        condition = None

        # center_coord used for flow matching
        center_coord = torch.stack(
            [
                uvgrids.coord[:, :, 0, 0],
                uvgrids.coord[:, :, 0, n_grid - 1],
                uvgrids.coord[:, :, n_grid - 1, 0],
                uvgrids.coord[:, :, n_grid - 1, n_grid - 1],
            ],
            dim=-1,
        ).mean(dim=-1)

        # Graph conditioning: encode topology + per-face attributes into a global z.
        condition = self.graph_encoder(
            face_adj=uvgrids.face_adj.float(),
        )

        if self.config.n_z > 1:
            condition = condition.repeat(1, self.config.n_z, 1)

        tokens = Tokens(
            sample=uvgrid_tensor,
            mask=uvgrid_empty_mask,
            condition=condition,
            center_coord=center_coord,
        )
        return tokens
