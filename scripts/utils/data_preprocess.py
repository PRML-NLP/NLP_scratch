import os
import sys

data_type = ''

if data_type == 'NMT':
    def read_data_nmt():
        with open(os.path.join('', '../data/kor.txt'), 'r') as f:
            return f.read()
        
    def preprocess_nmt(text):
        """Preprocess the English-French dataset."""
        def no_space(char, prev_char):
            return char in set(',.!?') and prev_char != ' '# 이전 단어가 빈칸이 아니고 지금 단어가 기호 이면
        words=''
        word = ''
        #text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
        text = text.lower()
        out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
        out = ''.join(out)
        #for sentence in text.split('\tcc-by 2 .0 (france) attribution: tatoeba .org #'):
        for sentence in out.split('\tcc-by 2 .0 (france) attribution: tatoeba .org #'):
            if '\n' in sentence:
                num = sentence.index('\n')
                word = sentence[num:]
            words += word
        return words

    def tokenize_nmt(text, num_examples=None):
        source, target = [], []
        for i, line in enumerate(text.split('\n')):
            if num_examples and i > num_examples:
                break
            word_lst = line.split('\t')
            if len(word_lst) == 2:
                source.append(word_lst[0].split(' '))
                target.append(word_lst[1].split(' '))
        return source, target