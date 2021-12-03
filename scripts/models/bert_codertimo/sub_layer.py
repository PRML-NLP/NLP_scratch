import torch
import torch.nn as nn


class FeedForwardLayer(nn.Module):
    """
    Description:
        ffnn(x) = max(0, x * w1 + b1) * w2 + b2
        f1 = x * w1 + b1
        f2 = gelu(0, f1)
        f3 = ffnn = f2 * w2 + b2

    Arguments:
        d_model: dimension of input and output layer
        d_ff: dimension of feed forward
    """

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.activ_func = nn.GELU()
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # input >> w_1 >> gelu >> dropout >> w_2 >> output
        return self.w_2(self.dropout(self.activ_func(self.w_1(x))))


class LayerNormalization(nn.Module):
    """
    Description:
        layer normalization = layernorm(x + sublayer(x))
        layernorm(x) = gamma * (x - mean(x)) / sqrt(std ** 2 + epsilon) + beta, gamma and beta are learnable=
        get mean and std over the feature dimension of input tensor
    """

    def __init__(self, d_feature: tuple, eps=1e-9):
        super().__init__()
        # assign learnable parameters, gamma and beta as feature dimension
        self.ln_gamma = nn.Parameter(torch.ones(d_feature))
        self.ln_beta = nn.Parameter(torch.zeros(d_feature))
        self.eps = eps

    def forward(self, x):
        # output dimension equals to feature dimension
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        # self.eps is broadcasted automatically to std tensor
        return (
            self.ln_gamma * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.ln_beta
        )


class NormalizedResidualConnection(nn.Module):
    """
    Description:
        h(x) = x + f(x)
    """

    def __init__(self, d_model, dropout):
        super().__init__()
        self.layer_norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sub_layer):
        # input >> normalization >> sub layer >> dropout >> + x >> output
        # return x + self.dropout(sub_layer(self.layer_norm(x)))
        return x + self.dropout(self.layer_norm.forward(sub_layer(x)))
