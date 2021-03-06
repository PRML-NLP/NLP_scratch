from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F
from torch.autograd import Variable

def init_lstm_weights(lstm, rand_unif_init_mag):
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
                
def init_linear_weights(linear, trunc_norm_init_std):
    linear.weight.data.normal_(std=trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=trunc_norm_init_std)

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, trunc_norm_init_std=1e-4, rand_unif_init_mag=0.02):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        # Vocab embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # Bidirectional LSTM
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.W_h = nn.Linear(hidden_dim*2, hidden_dim*2, bias=False)
                
        # Weights normalization
        self.embedding.weight.data.normal_(std=trunc_norm_init_std)
        init_linear_weights(self.W_h, trunc_norm_init_std)
        init_lstm_weights(self.lstm, rand_unif_init_mag)
            
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
        h, hidden_state = self.lstm(embedded)
                
        h, _ = pad_packed_sequence(h, batch_first=True)
        h = h.contiguous()
        
        feat = h.view(-1,self.hidden_dim*2)
        feat = self.W_h(feat).view(-1, h.size(1),self.hidden_dim*2)
               
        return h, feat, hidden_state


class Attention(nn.Module):
    def __init__(self, hidden_dim, trunc_norm_init_std=1e-4):
        super(Attention, self).__init__()
        self.v = nn.Linear(hidden_dim*2, 1, bias=False)
        self.W_c = nn.Linear(1, hidden_dim*2, bias=False)
        self.dec_proj = nn.Linear(hidden_dim*2, hidden_dim*2)
                        
        # weight initialization
        init_linear_weights(self.v, trunc_norm_init_std)
        init_linear_weights(self.W_c, trunc_norm_init_std)
        init_linear_weights(self.dec_proj, trunc_norm_init_std)
        
    def forward(self, h, enc_feat, attn_mask, s_t_hat, coverage):
        B, L, N = list(h.size())
        
        s_t_hat = self.dec_proj(s_t_hat)
        s_t_hat = s_t_hat.unsqueeze(1).expand(B, L, N).contiguous() # B, N -> B, L, N
        
        # e^t_i = v^T tanh(W_hxh_i + W_sxs_t + W_cxc^t_i + b_attn)
        attn_feat = enc_feat + s_t_hat + self.W_c(coverage.unsqueeze(-1))
        attn_feat = torch.tanh(attn_feat)
        attn_feat = self.v(attn_feat).view(-1, L)  # B x L
        attn_dist = F.softmax(attn_feat, dim=1)*attn_mask  # attention distribution a^t
        norm_factor = attn_dist.sum(1, keepdim=True)
        attn_dist = attn_dist / (norm_factor+1e-12)
        
        # Context vector
        context = torch.bmm(attn_dist.unsqueeze(1), h) # (B x 1 x L) * (B x L x N) => (B x 1 x N)
        context = context.squeeze(1)
        
        # Coverage accumulation
        coverage = coverage + attn_dist # B x L
        
        return attn_dist, context, coverage
    
    
class ReduceState(nn.Module):
    def __init__(self, hidden_dim, trunc_norm_init_std=1e-4):
        super(ReduceState, self).__init__()
        self.hidden_dim = hidden_dim        
        self.reduce_h = nn.Linear(hidden_dim*2, hidden_dim)
        self.reduce_c = nn.Linear(hidden_dim*2, hidden_dim)
        
        # weight initialization
        init_linear_weights(self.reduce_h, trunc_norm_init_std)
        init_linear_weights(self.reduce_c, trunc_norm_init_std)
        
    def forward(self, hidden_state):
        """hidden_state (tuple): (hidden_state, cell_state) [n_layers*D x B x hidden_dim]"""
        h, c = hidden_state  # 2 x B x hidden_dim
        
        # 2 x B x hidden_dim => B x 2 x hidden_dim => B x 2*hidden_dim
        h = h.transpose(0, 1).contiguous().view(-1, 2*self.hidden_dim) 
        c = c.transpose(0, 1).contiguous().view(-1, 2*self.hidden_dim)
        
        h = F.relu(self.reduce_h(h))
        c = F.relu(self.reduce_c(c))
        
        return (h.unsqueeze(0), c.unsqueeze(0)) # 1 x B x hidden_dim
                
        
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, trunc_norm_init_std=1e-4, rand_unif_init_mag=0.02):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        # Vocab embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # Unidirectional
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=False)
        # Attention
        self.attention = Attention(hidden_dim)
        
        self.feeding = nn.Linear(emb_dim+hidden_dim*2, emb_dim)
        self.w_p_gen = nn.Linear(hidden_dim*4+emb_dim, 1)
        self.V = nn.Linear(hidden_dim*3, hidden_dim)
        self.V_prime = nn.Linear(hidden_dim, vocab_size)
        
        # Weight initialization
        self.embedding.weight.data.normal_(std=trunc_norm_init_std)
        init_linear_weights(self.feeding, trunc_norm_init_std)
        init_linear_weights(self.w_p_gen, trunc_norm_init_std)
        init_linear_weights(self.V, trunc_norm_init_std)
        init_linear_weights(self.V_prime, trunc_norm_init_std)
        init_lstm_weights(self.lstm, rand_unif_init_mag)
                        
    def forward(self, y_t_prev, s_t_prev, h, enc_feat, c_t_pre, coverage, source_extend_vocab, enc_mask, extra_zeros=None):
        embedded = self.embedding(y_t_prev)
        
        # h_pre,c_pre = s_t_prev
        # s_t_hat = torch.cat((h_pre.view(-1, self.hidden_dim), c_pre.view(-1, self.hidden_dim)), 1) # B x 2*hidden_dim
        # attn_dist, context, coverage_next = self.attention(h, s_t_hat, coverage)
        
        # Input feeding
        embedded = self.feeding(torch.cat([embedded, c_t_pre],dim=-1))
        s_t_tilde, s_t = self.lstm(embedded.unsqueeze(1), s_t_prev)
        
        # output??? cell state??? ?????? ?????? 
        d_h, d_c = s_t
        s_t_hat = torch.cat((d_h.view(-1, self.hidden_dim), d_c.view(-1, self.hidden_dim)), 1) # B x 2*hidden_dim
        attn_dist, c_t, coverage_next = self.attention(h, enc_feat, enc_mask, s_t_hat, coverage)
        
        p_gen = torch.sigmoid(self.w_p_gen(torch.cat([c_t, s_t_hat, embedded], dim=-1)))
        P_vocab = F.softmax(self.V_prime(self.V(torch.cat([s_t_tilde.squeeze(1), c_t], dim=-1))),dim=1)
        
        P_vocab = p_gen*P_vocab
        attn_dist_ = (1.-p_gen)*attn_dist
        if extra_zeros is not None:
            P_vocab = torch.cat([P_vocab, extra_zeros], dim=-1)
        # if w is oov, largest attn_dist value is added to extended vocab dim
        P_w = P_vocab.scatter_add(1, source_extend_vocab, attn_dist_)  # vocab + oov
        
        return P_w, s_t, attn_dist, coverage_next
      

class PointerGenerator(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, max_dec_len, enc_n_layers=1, dec_n_layers=1, max_oovs=0, pad_id=0):
        super(PointerGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(vocab_size, emb_dim, hidden_dim, enc_n_layers)
        self.decoder = Decoder(vocab_size, emb_dim, hidden_dim, dec_n_layers)
        self.reducer = ReduceState(hidden_dim)
        self.max_dec_len = max_dec_len
        self.max_oovs = max_oovs
        self.pad_id = pad_id
    
    def forward(self, batch, cov_coef=1.0):
        source, source_extend_vocab, dec_input, target, enc_mask, dec_mask = batch
        
        src_lens = [seq.size(0) for seq in source]
        source = pad_sequence(source, batch_first=True) # by unk token
        h, enc_feat, enc_hidden_state = self.encoder(source, src_lens)
        s_t_prev = self.reducer(enc_hidden_state) # enc_hidden, enc_cell
        
        # init coverage
        c_t_pre = h.new(h.size(0), 2*self.hidden_dim).zero_()
        coverage = h.new(h.size(0), h.size(1)).zero_()
        extra_zeros = None
        if self.max_oovs > 0:
            extra_zeros = Variable(torch.zeros(h.size(0), self.max_oovs))
            if h.get_device() >= 0:
                extra_zeros = extra_zeros.to(h.get_device())
        
        step_losses = []
        for step in range(self.max_dec_len):
            y_t_pre = dec_input[:,step]
            # Decoding
            P_w, s_t_prev, attn_dist, next_coverage = self.decoder(y_t_pre, s_t_prev, h, enc_feat, c_t_pre, \
                coverage, source_extend_vocab, enc_mask, extra_zeros)
            # Get probability corresponding to target label
            target_t = target[:,step]
            P_w = torch.gather(P_w, 1, target_t.unsqueeze(1)).squeeze()
            covloss = torch.sum(torch.min(attn_dist, coverage), dim=1)
            lm_loss = -torch.log(P_w + 1e-12)
            step_losses.append(lm_loss + cov_coef*covloss)
            coverage = next_coverage
        
        loss = torch.sum(torch.stack(step_losses, 1), 1) # B
        
        return torch.mean(loss / (torch.sum(dec_mask, dim=-1)+1e-12))
    
    def generate(self, source):
        pass

        
if __name__=='__main__':
    x = torch.LongTensor([[1,2,3],[2,4,5],[2,1,3],[4,1,2]]) # 4 x 3 - B x L
    y = torch.LongTensor([[1,2,3],[2,4,5],[2,1,3],[4,1,2]]) # 4 x 3 - B x L
    x_m = torch.LongTensor([[1,1,0],[1,1,1],[1,1,0],[1,1,1]]) # 4 x 3 - B x L
    y_m = torch.LongTensor([[1,1,0],[1,1,1],[1,1,0],[1,1,1]]) # 4 x 3 - B x L
    model = PointerGenerator(100, emb_dim=40, hidden_dim=60, max_dec_len=3)
    print(model((x, x, y, y, x_m, y_m)))
    
    