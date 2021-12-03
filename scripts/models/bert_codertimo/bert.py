import torch.nn as nn
from .embedding import BertEmbedding
import torch
import torch.nn as nn
from .attention import MaskedMultiHeadAttention
from .sub_layer import FeedForwardLayer, NormalizedResidualConnection


class Encoder(nn.Module):
    """
    Description:
        encoder = multi-head attention + feed-forward layer + residual connection + layer normalization

    Arguments:
        d_model: dimension of input and output layer
        n_head: number of attention heads
        d_ff_hidden: dimension of feed-forward layer
    """

    def __init__(self, d_model, n_head, d_ff_hidden, dropout: float):
        super().__init__()
        self.attn = MaskedMultiHeadAttention(n_head=n_head, d_model=d_model)
        self.add_norm_layer = NormalizedResidualConnection(
            d_model=d_model, dropout=dropout
        )
        self.ff = FeedForwardLayer(d_model=d_model, d_ff=d_ff_hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask: torch.Tensor):
        y = self.add_norm_layer.forward(
            x, sub_layer=self.attn.forward(query=x, key=x, value=x, mask=mask)
        )
        z = self.add_norm_layer.forward(y, sub_layer=self.ff.forward(y))

        return self.dropout(z)


class Bert(nn.Module):
    """
    Description:
        making BERT architecture with transformer encoder blocks

    Arguments:
        n_vocab: number of vocabulary for embedding
        d_hidden: dimension of hidden layer
        n_layers: number of encoder blocks
        n_heads: number of attention heads
        dropout: ratio of unused nodes
    """

    def __init__(self, n_vocab, d_hidden=768, n_layers=12, n_heads=12, dropout=0.1):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.n_heads = n_heads
        # original BERT paper set feed forward dimension as hidden layer dimension * 4
        self.d_ff_hidden = d_hidden * 4
        self.embedding = BertEmbedding(n_vocab=n_vocab, d_embed=d_hidden)
        for _ in range(n_layers):
            self.enc_block = nn.ModuleList(
                [
                    Encoder(
                        d_hidden=d_hidden,
                        n_heads=n_heads,
                        d_ff_hidden=self.d_ff_hidden,
                        dropout=dropout,
                    )
                ]
            )

    def forward(self, x, seg_info):
        mask = (x > 0).unsqueeze(dim=1).repeat(1, x.size(1), 1).unsqueeze(dim=1)
        x = self.embedding(x, seg_info)

        for enc in self.enc_block:
            x = enc.forward(x, mask)

        return x
