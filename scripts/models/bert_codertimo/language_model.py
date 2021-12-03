import torch.nn as nn
from .bert import Bert


class BertLanguageModel(nn.Module):
    """
    Description:
        BERT language model = next sentence prediction + masked language model

    Arguments:
        bert: BERT model to train
        n_vocab: total size of vocabulary for masked laguage model
    """

    def __init__(self, bert: Bert, n_vocab):
        super().__init__()
        self.bert = bert
        self.nsp = NextSentencePrediction(d_hidden=self.bert.d_hidden)
        self.mlm = MaskedLanguageModel(d_hidden=self.bert.d_hidden, n_vocab=n_vocab)

    def forward(self, x, seg_label):
        x = self.bert(x, seg_label)

        return self.nsp(x), self.mlm(x)


class NextSentencePrediction(nn.Module):
    """
    Description:
        binary classification for (is_next, is_not_next)

    Arguements:
        d_hidden: output size of BERT model
    """

    def __init__(self, d_hidden):
        super().__init__()
        self.linear = nn.Linear(in_features=d_hidden, out_features=2)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.log_softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    Desription:
        prediction origin token with masked input tokenized sentence
        n-class classification for n_vocab (n_vocab = total size of vocabulary)
    """

    def __init__(self, d_hidden, n_vocab):
        super().__init__()
        self.linear = nn.Linear(in_features=d_hidden, out_features=n_vocab)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.log_softmax(self.linear(x))
