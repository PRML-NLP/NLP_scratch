from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F

def initialize_lstm_weights(lstm, rand_unif_init_mag):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-rand_unif_init_mag, rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, trunc_norm_init_std=1e-4, rand_unif_init_mag=0.02):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        # Vocab embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # Bidirectional LSTM
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
                
        # Weights normalization
        self.embedding.weight.data.normal_(std=trunc_norm_init_std)
        initialize_lstm_weights(self.lstm, rand_unif_init_mag)
            
    def forward(self, source, seq_lens):
        """return encoder hidden state h_i and decoder input

        Args:
            source (Tensor): Padded sequence tensor [Batch_size x Seq_len]
            seq_len (List): Original sequence length for each sample [Batch_size]
        Outputs:
            h (Tensor): [B x L x hidden_dim*D]
            hidden_state (tuple): (hidden_state, cell_state) [n_layers*D x B x hidden_dim]
        """
        embedded = self.embedding(source)
        """
        seq = torch.tensor([[1,2,0], [3,0,0], [4,5,6]])
        lens = [2, 1, 3]
        PackedSequence(data=tensor([4, 1, 3, 5, 2, 6]), batch_sizes=tensor([3, 2, 1]),
               sorted_indices=tensor([2, 0, 1]), unsorted_indices=tensor([1, 2, 0]))
        """
        embedded = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        # 
        h, hidden_state = self.lstm(embedded)
                
        h, _ = pad_packed_sequence(h, batch_first=True)
        h = h.contiguous()        
               
        return h, hidden_state


class Attention(nn.Module):
    def __init__(self, hidden_dim, trunc_norm_init_std=1e-4):
        self.v = nn.Linear(hidden_dim*2, 1, bias=False)
        self.W_h = nn.Linear(hidden_dim*2, hidden_dim*2, bias=False)
        self.W_s = nn.Linear(hidden_dim*2, hidden_dim*2, bias=False)
        self.W_c = nn.Linear(1, hidden_dim*2, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim*2))
        
        # weight initialization
        self.v.weight.data.normal_(std=trunc_norm_init_std)
        self.W_h.weight.data.normal_(std=trunc_norm_init_std)
        self.W_s.weight.data.normal_(std=trunc_norm_init_std)
        self.W_c.weight.data.normal_(std=trunc_norm_init_std)
        self.bias.weight.data.normal_(std=trunc_norm_init_std)
        start, end = hidden_dim // 2, hidden_dim
        self.bias.data.fill_(0.)
        self.bias.data[start:end].fill_(1.)
        
    def forward(self, h, s_t, coverage):
        B, L, N = list(h.size())
        s_t = s_t.unsqueeze(1).expand(B, L, N).contiguous() # B, N -> B, L, N
        
        attn_feat = self.W_h(h.view(-1, N))
        attn_feat = attn_feat + self.W_s(s_t.view(-1, N))
        attn_feat = attn_feat + self.W_c(coverage.view(-1, 1))
        
        # e^t_i = v^T tanh(W_hxh_i + W_sxs_t + W_cxc^t_i + b_attn)
        attn_feat = F.tanh(attn_feat + self.bias)
        attn_feat = self.v(attn_feat).view(-1, L) # B x L
        attn_dist = F.softmax(attn_feat, dim=1) # attention distribution a^t
        
        #### encoder input padding 고려해서 masking 구현해야 됨
        
        # Context vector
        context = torch.bmm(attn_dist.unsqueeze(1), h) # (B x 1 x L) * (B x L x N) => (B x 1 x N)
        context = context.squeeze(1)
        
        # Coverage accumulation
        coverage = coverage.view(B, L) + attn_dist
        
        return attn_dist, context, coverage
        
        

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, trunc_norm_init_std=1e-4, rand_unif_init_mag=0.02):
        # Vocab embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # Unidirectional
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=False)
        # Attention
        self.attention = Attention(hidden_dim)
        
        # Weight initialization
        self.embedding.weight.data.normal_(std=trunc_norm_init_std)
        initialize_lstm_weights(self.lstm, rand_unif_init_mag)
                        
    def forward(self, h, h_w, enc_hidden_state):
        pass
        

class ReduceState(nn.Module):
    """Reduce hidden state size to half"""
    
    def __init__(self, hidden_dim, trunc_norm_init_std=1e-4):
        super(ReduceState, self).__init__()
        
        self.reduce_h = nn.Linear(hidden_dim*2, hidden_dim)
        self.reduce_c = nn.Linear(hidden_dim*2, hidden_dim)
        
        # weight initialization
        self.reduce_h.weight.data.normal_(std=trunc_norm_init_std)
        self.reduce_c.weight.data.normal_(std=trunc_norm_init_std)
        
    def forward(self, hidden_state):
        pass
        

class PointerGenerator(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, enc_n_layers=1, dec_n_layers=1):
        self.encoder = Encoder(vocab_size, emb_dim, hidden_dim, enc_n_layers)
        self.decoder = Decoder(hidden_dim, dec_n_layers)
    
    def forward(self, source, target):
        src_lens = [seq.size(0) for seq in source]
        source = pad_sequence(source, batch_first=True)
        h, h_w, enc_hidden_state = self.encoder(source, src_lens)
        self.decoder(h, h_w, enc_hidden_state)
        
        
if __name__=='__main__':
    x = torch.LongTensor([[1,2,3],[3,4,5],[2,1,3],[4,1,2]]) # 4 x 3 - B x L
    encoder = Encoder(200, 50, 60, 3)
    encoder(x, [3,3,3,3])