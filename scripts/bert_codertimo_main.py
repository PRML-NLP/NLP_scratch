import argparse
from torch.utils.data import DataLoader
from .data.dataset import BertDataset
from .data.vocab import SubwordVocab
from .models.bert import Bert
from .trainer import BertTrainer


def bert_train():
    # create instance to input arguments
    parser = argparse.ArgumentParser(description="Arguments for training BERT")

    # set input arguments
    # dataset and path
    parser.add_argument(
        "-c", "--train_dataset", required=True, type=str, help="train dataset"
    )
    parser.add_argument(
        "-t", "--test_dataset", type=str, default=None, help="test dataset"
    )
    parser.add_argument(
        "-v", "--vocab_path", required=True, type=str, help="built vocab path"
    )
    parser.add_argument(
        "-o", "--output_path", required=True, type=str, help="path to save the model"
    )

    # model size
    parser.add_argument(
        "-hs",
        "--d_hidden",
        type=int,
        default=256,
        help="dimension of hidden layer of transformer module",
    )
    parser.add_argument(
        "-l", "--n_layers", type=int, default=8, help="number of layers"
    )
    parser.add_argument(
        "-a",
        "--n_heads",
        type=int,
        default=8,
        help="number of attention heads of transformer module",
    )
    parser.add_argument(
        "-s", "--seq_len", type=int, default=20, help="maximum input sequence length"
    )

    # trainer configuration
    parser.add_argument(
        "-b", "--batch_size", type=int, default=64, help="number of batch size"
    )
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument(
        "-w", "--num_workers", type=int, default=5, help="number of dataloader worker"
    )
    parser.add_argument(
        "--with_cuda", type=bool, default=True, help="whether training with CUDA or not"
    )
    parser.add_argument(
        "--log_freq", type=int, default=10, help="recording loss every n iterations"
    )
    parser.add_argument(
        "--corpus_lines", type=int, default=None, help="total number of lines in corpus"
    )
    # nargs=2 : it requires fixed value
    # nargs=? : 0 or 1
    # nargs=+ : it requires at least 1
    # nargs=* : it requires at least 0
    parser.add_argument(
        "--cuda_devices", type=int, nargs="+", default=None, help="CUDA device ids"
    )
    parser.add_argument(
        "--on_memory", type=bool, default=True, help="whether loading on memory or not"
    )

    # opimizer configuration
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate of optimizer"
    )
    parser.add_argument(
        "--optim_weight_decay",
        type=float,
        default=0.01,
        help="weight decaying value of optimizer",
    )
    parser.add_argument(
        "--optim_beta1", type=float, default=0.9, help="first beta value of optimizer"
    )
    parser.add_argument(
        "--optim_beta2",
        type=float,
        default=0.999,
        help="second beta value of optimizer",
    )

    # set input arguments as args variable
    args = parser.parse_args()

    # load vocabulary
    print("Loading vocab", args.vocab_path)
    vocab = SubwordVocab.load_vocab(vocab_path=args.vocab_path)
    print("Vocab size:", len(vocab))

    # load train dataset
    print("Loading train dataset", args.train_dataset)
    train_dataset = BertDataset(
        corpus_path=args.train_dataset,
        vocab=vocab,
        seq_len=args.seq_len,
        corpus_lines=args.corpus_lines,
        on_memory=args.on_memory,
    )

    # load test dataset if it exists
    print("Loading test dataset", args.test_dataset)
    if args.test_dataset is not None:
        test_dataset = BertDataset(
            corpus_path=args.test_dataset,
            vocab=vocab,
            seq_len=args.seq_len,
            on_memory=args.on_memory,
        )

    # create dataloaders for train and test dataset
    print("Creating dataloader")
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    if test_dataset is not None:
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    # build BERT model
    print("Building BERT model")
    bert = Bert(
        n_vocab=len(vocab),
        d_hidden=args.d_hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )

    # get trainer of BERT model
    print("Getting BERT trainer")
    trainer = BertTrainer(
        bert=bert,
        n_vocab=len(vocab),
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        lr=args.lr,
        betas=(args.optim_beta1, args.optim_beta2),
        weight_decay=args.optim_weight_decay,
        cuda_devices=args.cuda_devices,
        log_freq=args.log_freq,
    )

    # start training
    print("Start training")
    for epoch in range(args.epochs):
        trainer.train_model(epoch=epoch)
        trainer.save_model(epoch=epoch, file_path=args.output_path)

        if test_dataloader is not None:
            trainer.test_model(epoch=epoch)
