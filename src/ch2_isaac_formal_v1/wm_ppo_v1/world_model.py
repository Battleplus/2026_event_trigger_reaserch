"""Event-aware graph/GRU model contract.

The model predicts a real one-step dynamic target and an optional paired gain
target.  It does not infer a one-step label from a 30-minute endpoint.
"""

from __future__ import annotations

from typing import Any

from .features import EVENT_DIM

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - dependency is checked by the caller
    torch = None
    Tensor = Any
    nn = object  # type: ignore[assignment]


if torch is not None:

    class EventAwareWorldModel(nn.Module):
        """Shared node encoder, masked graph pool, one-step GRU and heads."""

        def __init__(self, node_dim: int, hidden_dim: int = 64, dynamic_dim: int = 4,
                     event_dim: int = EVENT_DIM, action_dim: int = 5) -> None:
            super().__init__()
            self.node_encoder = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.ReLU())
            self.message = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.edge_embedding = nn.Embedding(3, hidden_dim)
            self.graph_update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU())
            self.event_encoder = nn.Sequential(nn.Linear(event_dim, hidden_dim), nn.ReLU())
            self.history_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.dynamic_head = nn.Linear(hidden_dim + action_dim, dynamic_dim)
            self.gain_head = nn.Linear(hidden_dim, 1)

        def forward(self, node_features: Tensor, node_mask: Tensor, sequence_mask: Tensor,
                    edge_index: Tensor | None = None, edge_mask: Tensor | None = None,
                    edge_type: Tensor | None = None, event_features: Tensor | None = None,
                    action_features: Tensor | None = None) -> dict[str, Tensor]:
            if node_features.ndim != 4 or node_mask.ndim != 3 or sequence_mask.ndim != 2:
                raise ValueError("expected [batch,time,nodes,features], node mask, and sequence mask")
            encoded = self.node_encoder(node_features)
            if edge_index is not None:
                if edge_mask is None or edge_type is None or edge_index.ndim != 4:
                    raise ValueError("edge_index requires edge_mask, edge_type, and [batch,time,2,edges]")
                bsz, steps, nodes, width = encoded.shape
                if edge_index.shape[:2] != (bsz, steps) or edge_index.shape[2] != 2:
                    raise ValueError("edge tensors do not match the node batch")
                if edge_mask.shape != edge_type.shape or edge_mask.shape != edge_index.shape[:2] + edge_index.shape[3:]:
                    raise ValueError("edge masks and types must match padded edge dimensions")
                if edge_mask.any():
                    active = edge_mask.bool()
                    src, dst = edge_index[:, :, 0], edge_index[:, :, 1]
                    if ((src[active] < 0) | (src[active] >= nodes) | (dst[active] < 0) | (dst[active] >= nodes)).any():
                        raise ValueError("active edge endpoint exceeds padded node dimension")
                    flat = encoded.reshape(bsz * steps, nodes, width)
                    src_flat = src.reshape(bsz * steps, -1)
                    dst_flat = dst.reshape(bsz * steps, -1)
                    valid = active.reshape(bsz * steps, -1)
                    flat_node_mask = node_mask.reshape(bsz * steps, nodes).bool()
                    typ = edge_type.reshape(bsz * steps, -1).clamp(0, 2)
                    batch_ids = torch.arange(bsz * steps, device=encoded.device).unsqueeze(1).expand_as(src_flat)
                    valid = valid & flat_node_mask[batch_ids, src_flat] & flat_node_mask[batch_ids, dst_flat]
                    messages = self.message(flat[batch_ids, src_flat]) + self.edge_embedding(typ)
                    messages = messages * valid.unsqueeze(-1).to(messages.dtype)
                    aggregate = torch.zeros_like(flat).index_put_((batch_ids[valid], dst_flat[valid]), messages[valid], accumulate=True)
                    degree = torch.zeros((bsz * steps, nodes, 1), dtype=encoded.dtype, device=encoded.device)
                    degree.index_put_((batch_ids[valid], dst_flat[valid]), torch.ones_like(messages[valid, :1]), accumulate=True)
                    aggregate = aggregate / degree.clamp_min(1.0)
                    encoded = self.graph_update(torch.cat((flat, aggregate), dim=-1)).reshape(bsz, steps, nodes, width)
                    encoded = encoded * node_mask.unsqueeze(-1).to(encoded.dtype)
            mask = node_mask.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
            if event_features is not None:
                pooled = pooled + self.event_encoder(event_features)
            output, _ = self.history_gru(pooled)
            lengths = sequence_mask.to(torch.long).sum(dim=1).clamp_min(1) - 1
            final = output[torch.arange(output.shape[0], device=output.device), lengths]
            if action_features is None:
                action_final = torch.zeros((final.shape[0], self.dynamic_head.in_features - final.shape[1]),
                                           dtype=final.dtype, device=final.device)
            else:
                action_final = action_features[torch.arange(output.shape[0], device=output.device), lengths]
            return {"dynamic": self.dynamic_head(torch.cat((final, action_final), dim=-1)),
                    "gain": self.gain_head(final).squeeze(-1)}

else:

    class EventAwareWorldModel:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is required for EventAwareWorldModel; CPU fallback is disabled")


def require_torch() -> Any:
    """Return torch or raise an explicit dependency status."""

    if torch is None:
        raise RuntimeError("PYTORCH_UNAVAILABLE")
    return torch
