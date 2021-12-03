import torch
from torch import nn
import sys
import os
from torch._C import dtype
from torch.utils import data
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.data_preprocess import read_data_nmt, preprocess_nmt, tokenize_nmt
from data.NMT_data.vocabulary import Vocab_nmt
from models.encoder_decoder_gru import src_vocab,tgt_vocab,trainer,nmt
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl

num_steps = 10
engs = input('>>> ')
def word2vec(sentence,vocab,num_steps):
    vocab= vocab
    sentence = sentence.lower().split(' ')
    word_emb = [src_vocab['<pad>']]*num_steps
    for index,value in enumerate(sentence):
        word_emb[index] = vocab[value]
    word_emb[index+1] = vocab['<eos>']
    return torch.tensor([word_emb])

def translation(prediction,vocab):
    tgt_vocab = vocab
    tgt_vocab_rvs = {}
    for key, value in tgt_vocab.items():
        tgt_vocab_rvs[value] = key
    pred = prediction
    ans = []
    for word in pred[0]:
        words = []
        for token in word:
            words.append(tgt_vocab_rvs[int(token)])
        ans.append(words)
    return ans


input = word2vec(engs,src_vocab,num_steps)
pred_iter = data.DataLoader((input),len(engs),shuffle=False)
pred_= trainer.predict(nmt,pred_iter)
answer = translation(pred_,tgt_vocab)
print(">>>>",answer)