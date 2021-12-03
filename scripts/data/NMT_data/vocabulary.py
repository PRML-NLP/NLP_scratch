import os
import collections

class Vocab_nmt:
    def __init__(self,tokens,threshold,special_tokens=[]):
        self.threshold = threshold
        self.special_tokens = special_tokens
        self.tokens = tokens
        self.token_lst = [token for turn in self.tokens for token in turn]
        self.counter = self.count_frequency()
        self.vocab_lst = self.voc_lst()
        self.vocab_idx = self.voc_idx()
    
    """Counting frequency of words"""    
    def count_frequency(self):
        return collections.Counter(self.token_lst)
    """Combining special tokens and words """
    def voc_lst(self):
        vocab_dict = list(['<unk>'] + self.special_tokens + sorted(set( [token for token, freq in self.counter.items() if freq >= self.threshold])))
        return vocab_dict
    """Generating word dictionary"""
    def voc_idx(self):
        return {value : idx for idx, value in enumerate(self.vocab_lst)}