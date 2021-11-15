import torch
from torch import nn
import sys
import os
from torch.utils import data
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.data_preprocess import read_data_nmt, preprocess_nmt, tokenize_nmt
from data.Vocabulary import Vocab_nmt

"""Embedding words"""

def word_embedding_nmt(sentence:list,vocab, emb_dim, padding_token):
    if len(sentence) > emb_dim:
        return sentence[:emb_dim]
    word_emb = [vocab[padding_token]]*emb_dim
    for index,value in enumerate(sentence):
        word_emb[index] = vocab[value]
    word_emb[index+1] = vocab['<eos>']
    return word_emb

def build_array_nmt(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches."""
    array = torch.tensor([word_embedding_nmt(l,vocab, num_steps, '<pad>') for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def load_data_nmt(batch_size, num_steps, num_examples=600):
    """Return the iterator and the vocabularies of the translation dataset."""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab_nmt(source,0,['<pad>', '<bos>','<eos>']).vocab_idx
    tgt_vocab = Vocab_nmt(target,0,['<pad>', '<bos>','<eos>']).vocab_idx
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


"""Encoder/Decoder model"""
class Encoder_nmt(nn.Module):
    def __init__(self, vocab_size, embed_dimn, num_hiddens, num_layers,dropout:float=0.1):
        
        # Initializing Encoder_nmt
        super(Encoder_nmt, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dimn)# in vocabulary, we represent all words by Embedding layer in embed_dimension
        self.rnn = nn.GRU(embed_dimn, num_hiddens, num_layers,dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        X = self.embedding(X).permute(1, 0, 2)
        X = X.permute(1, 0, 2)
        
        '''
        <1>
        i, am, a, boy, <pad>, <pad>, <pad>, <pad>
        you, are , a, girl, <pad>, <pad>, <pad>, <pad>
        we, are, the, one, <pad>, <pad>, <pad>, <pad>, <pad> 
        (batch_size,embedding_size) -> (3,10)
        
        <2>
        i, am, a, boy, <pad>, <pad>, <pad>, <pad> -> embedding dimension
        (batch_size, embedding_size, embedding_dimension(Embedding))
        
        <3>
        i, you, we 
        am ,are, are
        a, a, the 
        (embedding size, batch_size, embedding_dimension)
        '''
        output, state = self.rnn(X)
        state = self.dropout(state)
        # `output` shape: (`embedding size`, `batch_size`, `num_hiddens`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
    
    
class Decoder_nmt(nn.Module):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,dropout:float=0.1):
        super(Decoder_nmt, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)
        context = state[-1].repeat(X.shape[0], 1, 1)# Using hidden state from encoder's last layer
        X_and_context = torch.cat((X, context), 2)# (embed_size + num_hiddens)
        
        output, state = self.rnn(X_and_context, state)
        output = self.dropout(output)
        output = self.dense(output).permute(1, 0, 2)
        # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target):
        enc_outputs, dec_state = self.encoder(source)
        return self.decoder(target, dec_state)
## Set parameters ##

embedding_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs = 0.005, 300

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
encoder = Encoder_nmt(len(src_vocab), embedding_size, num_hiddens, num_layers,dropout)
decoder = Decoder_nmt(len(tgt_vocab), embedding_size, num_hiddens, num_layers,dropout)
net = EncoderDecoder(encoder, decoder)
print(net)
