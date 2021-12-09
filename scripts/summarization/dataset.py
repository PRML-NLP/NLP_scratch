import glob
import random
import struct
import csv, os
import torch
from torch.utils.data import DataLoader, Dataset
from nltk.tokenize import word_tokenize
import pickle, json
from tqdm import tqdm

class Vocab:
    def __init__(self, vocab_path=None):
        self.idx2item = {}
        self.item2idx = {}
        self.freq_dict = {}
        self.special_tokens = {
            'pad': '<pad>', 'unk': '<unk>', 'bos': '<s>', 'eos': '</s>'
        }
        
        if vocab_path is not None and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
        else:
            self.init()
            
    def init(self):
        for _, item in self.special_tokens.items():
            self.absolute_add_item(item)
        
    def __len__(self):
        return len(self.idx2item)
    
    def absolute_add_item(self, item):
        idx = len(self)
        self.idx2item[idx] = item
        self.item2idx[item] = idx
    
    def add_item(self, item):
        if item not in self.freq_dict:
            self.freq_dict[item] = 0
        self.freq_dict[item] += 1
    
    def construct(self, limit):
        l = sorted(self.freq_dict.keys(), key=lambda x: -self.freq_dict[x])
        print('Actual label size %d' % (len(l) + len(self.idx2item)))
        if len(l) + len(self.idx2item) < limit:
            print('actual vocab set smaller than that configured: {}/{}'
                            .format(len(l) + len(self.idx2item), limit))
        for item in l:
            if item not in self.item2idx:
                idx = len(self.idx2item)
                self.idx2item[idx] = item
                self.item2idx[item] = idx
                if len(self.idx2item) >= limit:
                    break

    def encode(self, item):
        if not item in self.item2idx:
            return self.item2idx[self.special_tokens['unk']]
        return self.item2idx[item]

    def decode(self, idx):
        return self.idx2item[idx]

    def load_vocab(self, vocab_path):
        f = open(vocab_path, 'rb')
        dic = pickle.load(f)
        self.idx2item = dic['idx2item']
        self.item2idx = dic['item2idx']
        self.freq_dict = dic['freq_dict']
        f.close()

    def save_vocab(self, vocab_path):
        f = open(vocab_path, 'wb')
        dic = {
            'idx2item': self.idx2item,
            'item2idx': self.item2idx,
            'freq_dict': self.freq_dict
        }
        pickle.dump(dic, f)
        f.close()

    def sentence_encode(self, word_list):
        return [self.encode(_) for _ in word_list]

    def sentence_decode(self, index_list, eos=None):
        l = [self.decode(_) for _ in index_list]
        if not eos or eos not in l:
            return ' '.join(l)
        else:
            idx = l.index(eos)
            return ' '.join(l[:idx])

    def nl_decode(self, l, eos=None):
        return [self.sentence_decode(_, eos) + '\n' for _ in l]

    def encode(self, item):
        if item in self.item2idx:
            return self.item2idx[item]
        else:
            return self.item2idx['<unk>']

    def decode(self, idx):
        # if idx < len(self):
        return self.idx2item[idx]
        # else:
        #     return 'ITEM_%d' % (idx - self.vocab_size)


class CnnDailyMailDataset(Dataset):
    def __init__(self, data_dir, vocab_file=None, max_enc_len=400, max_dec_len=100, vocab_size=50000, mode='train', cached=False):                
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        
        if mode!='train' and (vocab_file is None or not os.path.exists(vocab_file)):
            raise ValueError('You need to construct vocab with training data.')
        
        self.vocab = Vocab(vocab_path=vocab_file)
        self.special_tokens = self.vocab.special_tokens
        
        if cached and vocab_file is not None \
            and os.path.exists(os.path.join(data_dir, mode+'.json')):
            print('Loading', os.path.join(data_dir, mode+'.json'),'...')
            with open(os.path.join(data_dir, mode+'.json'), 'r') as fin:
                data = json.load(fin)
                self.sources = data['sources']
                self.targets = data['targets']
                self.enc_masks = data['enc_masks']
                self.dec_masks = data['dec_masks']
                self.source_extend_vocabs = data['source_extend_vocabs']
                self.dec_inputs = data['dec_inputs']
                self.max_art_oovs = data['max_art_oovs']
        else:
            self.sources = []
            self.targets = []
            self.enc_masks = []
            self.dec_masks = []
            self.source_extend_vocabs = []
            self.dec_inputs = []
            
            with open(os.path.join(data_dir, mode+'.source'), 'r') as fin:
                sources = fin.readlines()
            with open(os.path.join(data_dir, mode+'.target'), 'r') as fin:
                targets = fin.readlines()
                
            assert len(self.sources)==len(self.targets)
            
            print('Tokenize and convert tokens to id...') 
            # detach 'not' because of vocab file is None
            construct_vocab = not (vocab_file is not None and os.path.exists(vocab_file))            
            if construct_vocab:
                print('There is not vocab. Vocab will be constructed with documents.')
            # Tokenize and construct vocab
            for target in tqdm(targets):
                self.targets.append(
                    self.get_tokenized_data(target, construct_vocab=construct_vocab))
            del targets
            for source in tqdm(sources):
                self.sources.append(
                    self.get_tokenized_data(source, construct_vocab=construct_vocab))
            del sources
            
            if construct_vocab:
                self.vocab.construct(vocab_size)
                self.vocab.save_vocab(vocab_file if vocab_file is not None else 'vocab.json')
                        
            pad_id = self.vocab.encode(self.special_tokens['pad'])
            
            print('Encoding articles and abstracts...')
            article_oovs_list = []
            for i in tqdm(range(len(self.sources))):
                enc_input_extend_vocab, article_oovs = self.article2ids(self.sources[i])
                source = self.vocab.sentence_encode(self.sources[i])
                # Encoder input
                if len(source) >= self.max_enc_len:
                    source = source[:self.max_enc_len]
                    enc_input_extend_vocab = enc_input_extend_vocab[:self.max_enc_len]
                    enc_mask = [1]*self.max_enc_len
                else:
                    source, enc_mask = self.make_input_id_mask(source, pad_id, self.max_enc_len)
                    enc_input_extend_vocab = self.make_input_id_mask(enc_input_extend_vocab, pad_id, self.max_enc_len, mask=False)
                self.sources[i] = source
                self.source_extend_vocabs.append(enc_input_extend_vocab.copy())
                self.enc_masks.append(enc_mask)
                # Decoder input
                dec_input, target = self.build_decoder_input_target(self.targets[i], article_oovs)
                if len(target) >= self.max_dec_len:
                    target = target[:self.max_dec_len]
                    dec_input = dec_input[:self.max_dec_len]
                    dec_mask = [1]*self.max_dec_len
                else:
                    target = self.make_input_id_mask(target, pad_id, self.max_dec_len, mask=False)
                    dec_input, dec_mask = self.make_input_id_mask(dec_input, pad_id, self.max_dec_len)
                self.targets[i] = target
                self.dec_inputs.append(dec_input.copy())
                self.dec_masks.append(dec_mask)
                article_oovs_list.append(len(article_oovs))
            
            self.max_art_oovs = max(article_oovs_list)
            
            data = {}                
            data['sources'] = self.sources
            data['source_extend_vocabs'] = self.source_extend_vocabs
            data['enc_masks'] = self.enc_masks
            data['targets'] = self.targets
            data['dec_inputs'] = self.dec_inputs
            data['dec_masks'] = self.dec_masks
            data['max_art_oovs'] = self.max_art_oovs
            
            print("Writing converted data...")
            with open(os.path.join(data_dir, mode+'.json'), 'w') as fout:
                json.dump(data, fout, ensure_ascii=True)
        
        assert len(self.sources)==len(self.targets)==len(self.source_extend_vocabs)==len(self.dec_inputs)
        
        print('The number of data:', len(self.sources))
    
    
    def build_decoder_input_target(self, tokenized_abs, article_oovs):
        bos = self.vocab.encode(self.special_tokens['bos'])
        eos = self.vocab.encode(self.special_tokens['eos'])
        abs_ids = self.vocab.sentence_encode(tokenized_abs)
        abs_ids_extend_vocab = self.abstract2ids(tokenized_abs, article_oovs)
        dec_input = [bos] + abs_ids
        target = abs_ids_extend_vocab
        if len(dec_input) > self.max_dec_len:
            dec_input = dec_input[:self.max_dec_len]
            target = target[:self.max_dec_len]
        else:
            target.append(eos)
        
        assert len(dec_input)==len(target)
        
        return dec_input, target
        
    
    def article2ids(self, tokenized_doc):
        ids = []
        oovs = []
        unk_id = self.vocab.encode(self.special_tokens['unk'])
        for w in tokenized_doc:
            i = self.vocab.encode(w)
            if i == unk_id:
                if w not in oovs:
                    oovs.append(w)
                oov_num = oovs.index(w)
                ids.append(len(self.vocab) + oov_num)
            else:
                ids.append(i)
        return ids, oovs
    
    def abstract2ids(self, tokenized_doc, article_oovs):
        ids = []
        unk_id = self.vocab.encode(self.special_tokens['unk'])
        for w in tokenized_doc:
            i = self.vocab.encode(w)
            if i==unk_id:
                if w in article_oovs:
                    vocab_idx = len(self.vocab) + article_oovs.index(w)
                    ids.append(vocab_idx)
                else:
                    ids.append(unk_id)
            else:
                ids.append(i)
        return ids
    
    def get_tokenized_data(self, raw_data, construct_vocab):
        tokenized = word_tokenize(raw_data)
        if construct_vocab:
            for token in tokenized:
                self.vocab.add_item(token)
        return tokenized
    
    @staticmethod
    def make_input_id_mask(input_id, pad_id, max_seq_len, mask=True):
        if mask:
            attention_mask = [1] * len(input_id) + [0]*(max_seq_len-len(input_id))
        input_id += [pad_id]*(max_seq_len-len(input_id))
        if not mask:
            return input_id
        
        return input_id, attention_mask
           
    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, item):
        return torch.LongTensor(self.sources[item]), torch.LongTensor(self.source_extend_vocabs[item]), \
               torch.LongTensor(self.dec_inputs[item]), torch.LongTensor(self.targets[item]), \
               torch.FloatTensor(self.enc_masks[item]), torch.FloatTensor(self.dec_masks[item])
               
if __name__=='__main__':
    CnnDailyMailDataset(data_dir='data/cnn_dm', vocab_file='data/vocab.pkl', mode='train')