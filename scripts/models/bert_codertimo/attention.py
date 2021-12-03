import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedAttention(nn.Module):
    """compute masked scaled dot product attention"""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value,
        mask: torch.Tensor = None,
        dropout=None,
    ):
        attn_score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            query.size(-1)
        )

        if mask is not None:
            # make element value to to negative 10^9 corresponding to position where element of mask tensor equals 0
            masked_attn_score = attn_score.masked_fill(mask == 0, value=-1e9)

        # set dim=-1 to apply softmax function for each row
        attn_prob = F.softmax(masked_attn_score, dim=-1)

        if dropout is not None:
            # dropout equals nn.Dropout(p)
            attn_prob = dropout(attn_prob)

        return torch.matmul(attn_prob, value), attn_prob


class MaskedMultiHeadAttention(nn.Module):
    """
    Description:
        compute masked multi-head scaled dot product attention

    Arguments:
        n_head: number of attention heads
        d_model: dimension of input and output layer
    """

    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()
        self.d_k = d_model // n_head
        self.n_head = n_head
        # nn.Linear() returns same dimension as input dimension
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # output weight for concatenated tensor
        self.w_o = nn.Linear(d_model, d_model)
        self.attn = MaskedAttention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask: torch.Tensor = None):
        # set batch size
        batch_size = query.size(0)

        # linear projection in batch from d_model => d_k * n_head
        query = (
            self.w_q(query).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        )
        key = self.w_k(key).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        value = (
            self.w_v(value).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        )

        # apply attention on all the projected tensors in batch
        ctx, _ = self.attn.forward(
            query=query, key=key, value=value, mask=mask, dropout=self.dropout
        )
        # get effect of concatenate using a view and apply a final linear
        # transpose() changes only indices of tensor, not components, so contiguous() is needed to prevent memory error
        # dimension of ctx equals to dimension of input embedding
        ctx = (
            ctx.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_head * self.d_k)
        )

        return self.w_o(ctx)
