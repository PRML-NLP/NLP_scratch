import random
from torch.utils.data import Dataset
from tqdm import tqdm


class BertDataset(Dataset):
    def __init__(
        self,
        corpus_path,
        vocab,
        seq_len,
        encoding="utf-8",
        corpus_lines=None,
        on_memory=True,
    ):
        super().__init__()
        self.corpus_path = corpus_path
        self.vocab = vocab
        self.seq_len = seq_len
        self.encoding = encoding
        self.corpus_lines = corpus_lines
        self.on_memory = on_memory

        with open(self.corpus_path, "r", encoding=self.encoding) as f:
            # in case that there is no corpus line information and also not on memory
            if self.corpus_lines is None and not self.on_memory:
                # load dataset
                for _ in tqdm(f, desc="Loading dataset", total=self.corpus_lines):
                    # count the corpus length
                    self.corpus_lines += 1
            # in case that corpus is on memory
            if self.on_memory:
                for line in tqdm(f, desc="Loading dataset", total=self.corpus_lines):
                    #
                    self.lines = [line[:-1].split("\t")]
                    # set corpus length as corpus length
                    self.corpus_lines = len(self.lines)

        # in case that corpus is not on memory
        if not self.on_memory:
            # ???????????????
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            # ???????????????
            self.random_file = open(self.corpus_path, "r", encoding=self.encoding)

            if self.corpus_lines < 1000:
                for _ in range(random.randint(0, self.corpus_lines)):
                    # take out one by one up to corpus lines length
                    self.random_file.__next__()
            else:
                for _ in range(random.randint(0, 1000)):
                    # take out one by one up to 1,000
                    self.random_file.__next__()
