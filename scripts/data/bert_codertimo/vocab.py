import argparse
import pickle
from collections import Counter
import torchtext
from torch.nn.modules.module import T
from tqdm import tqdm


class Vocab(torchtext.vocab.Vocab):
    def __init__(self, counter: Counter, max_vocab_size=None, min_freq=1):
        super().__init__(
            counter,
            specials=["<pad>", "<unk>", "<eos>", "<bos>", "<mask>"],
            max_size=max_vocab_size,
            min_freq=min_freq,
        )
        # set special token indices
        self.pad_idx = 0
        self.unk_idx = 1
        self.eos_idx = 2
        self.bos_idx = 3
        self.mask_idx = 4

    def load_vocab(self, vocab_path: str):
        with open(vocab_path, "rb") as f:

            return pickle.load(f)

    def save_vocab(self, vocab_path: str):
        with open(vocab_path, "wb") as f:

            # save overall Vocab class at vocabulary path
            return pickle.dump(self, f)


class SubwordVocab(Vocab):
    """build vocabulary with text corpus"""

    def __init__(self, texts, max_vocab_size=None, min_freq=1):
        # show the start of making vocabulary
        print("Build vocabulary")
        counter = Counter()

        for line in tqdm(texts):
            if isinstance(line, list):
                words = line
            else:
                # remove \n and \t and assign as list dtype
                words = line.replace("\n", "").replace("\t", "").split()

            for word in words:
                # add a frequency number corresponding to the key of the counter dictionary
                counter[word] += 1
        super().__init__(counter, max_size=max_vocab_size, min_freq=min_freq)

    def load_vocab(self, vocab_path: str):
        with open(vocab_path, "rb") as f:

            return pickle.load(f)


def build_vocab():
    # create instance to input arguments
    parser = argparse.ArgumentParser(description="Arguments for building vocabulary")

    # set input arguments
    parser.add_argument("-cp", "--corpus_path", required=True, type=str)
    parser.add_argument("-op", "--output_path", required=True, type=str)
    parser.add_argument("-vs", "--vocab_size", type=int, default=None)
    parser.add_argument("-enc", "--encoding", type=str, default="utf-8")
    parser.add_argument("-mf", "--min_freq", type=int, default=1)

    # set input arguments as args variable
    args = parser.parse_args()

    # create vocabulary with corpus data
    with open(args.corpus_path, "r", encoding=args.encoding) as f:
        vocab = SubwordVocab(
            texts=f, max_vocab_size=args.vocab_size, min_freq=args.min_freq
        )

    print("Vocab size:", len(vocab))
    # save vocabulary at output path
    vocab.save_vocab(args.output_path)
