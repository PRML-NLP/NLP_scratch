import torch
import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    """
    Description:
        inherit torch.nn.Embedding and use this as is
        declare this class here and do the actual embedding separately

    Arguments:
        n_vocab: number of vocabulary for embedding
        d_embed: dimension of expression a token in embedding vector
        padding_idx: index of token for padding
    """

    def __init__(self, n_vocab, d_embed=512):
        # override torch.nn.Embedding
        super().__init__(num_embeddigns=n_vocab, embedding_dim=d_embed, padding_idx=0)


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Arguments:
        d_model: dimension of input and output layer
        max_len: maximum length of input sequence
    """

    def __init__(self, d_model, max_len=512):
        # orverride torch.nn.Module
        super().__init__()

        # assign positional embedding with zeros
        pos_embed = torch.zeros(size=(max_len, d_model), dtype=torch.float)
        # prevent optimizer to update position embedding
        pos_embed.requires_grad = False

        # assign position and doubled index
        # pos : position of embedding vector of input sequence
        # _2i : 2 * (index of dimension of embedding vector)
        pos = torch.arange(start=0, end=max_len, dtype=torch.float).unsqueeze(dim=1)
        _2i = torch.arange(start=0, end=d_model, step=2, dtype=torch.float)

        # make position embedding with sin and cos func according to odd and even
        pos_embed[:, 0::2] = torch.sin(pos / torch.pow(10000, (_2i / d_model)))
        pos_embed[:, 1::2] = torch.cos(pos / torch.pow(10000, (_2i / d_model)))
        # change dimension of position embedding
        pos_embed = pos_embed.unsqueeze(dim=0)

        # prevent optimizer to update position embedding
        self.register_buffer(name="pos_embed", tensor=pos_embed)

    def forward(self, x):
        # return position embedding up to length of input sequence
        return self.pos_embed[:, : x.size(1)]


class SegmentEmbedding(nn.Embedding):
    """
    Description:
        inherit torch.nn.Embedding and use this as is
        declare this class here and do the actual embedding separately

    Arguments:
        d_embed: dimension of expression a token in embedding vector
    """

    def __init__(self, d_embed=512):
        # num_embedding : segment(2) + padding(1)
        super().__init__(num_embedding=3, embedding_dim=d_embed, padding_idx=0)


class BertEmbedding(nn.Module):
    def __init__(self, n_vocab, d_embed, dropout):
        super().__init__()
        self.tok_embed = TokenEmbedding(n_vocab=n_vocab, d_embed=d_embed)
        self.pos_embed = SinusoidalPositionalEmbedding(
            d_model=self.tok_embed.embedding_dim
        )
        self.seg_embed = SegmentEmbedding(d_embed=self.tok_embed.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, seg_label):
        y = self.tok_embed(x) + self.pos_embed(x) + self.seg_embed(seg_label)

        return self.dropout(y)
