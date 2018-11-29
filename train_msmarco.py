from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import random
from tqdm import tqdm, trange

import numpy as np
import torch

from pytorch_pretrained_bert.qa_modeling import MSmorco
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.data.datasets import make_msmarco
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.data.bert_dataset import MSmarco_iterator

def main():
    parser = argparse.ArgumentParser()


    parser.add_argument("--save-dir", default="checkpoints", type=str,
                        help="path to save checkpoints")

    ## Other parameters
    parser.add_argument("--data", default="data", type=str, help="msmorco train and dev data")
    parser.add_argument("--origin-data", default="data", type=str, help="msmorco train and dev data, will be tokenizer")
    parser.add_argument("--path", default="data", type=str, help="path(s) to model file(s), colon separated")
    parser.add_argument("--pre-dir", type=str,
                        help="where the pretrained checkpoint")
    parser.add_argument("--max-passage-tokens", default=200, type=int,
                        help="The maximum total input passage length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max-query-tokens", default=50, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--train-batch-size", default=1, type=int, help="Total batch size for training.")
    parser.add_argument("--predict-batch-size", default=1, type=int, help="Total batch size for predictions.")
    parser.add_argument("--lr", default=6.25e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num-train-epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup-proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient-accumulation-steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--do-lower-case',
                        default=False, action='store_true',
                        help='whether case sensitive')

    args = parser.parse_args()

    # first make corpus
    # tokenizer = BertTokenizer.build_tokenizer(args)
    # make_msmarco(args, tokenizer)

    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()


    print("| gpu count : {}".format(n_gpu))
    print("| train batch size in each gpu : {}".format(args.train_batch_size))
    print("| biuid tokenizer and model in : {}".format(args.pre_dir))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    tokenizer = BertTokenizer.build_tokenizer(args)
    data_iter = MSmarco_iterator(args, tokenizer, batch_size=args.train_batch_size, world_size=n_gpu, name="msmarco_train.pk")
    data_size = len(data_iter)
    num_train_steps = args.num_train_epochs*data_size//(args.train_batch_size*n_gpu)
    print("| load dataset {}".format(data_size))

    model = MSmorco.build_model(args)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.lr,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    model.train()
    ok = 0
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(data_iter, desc="Iteration")):
            loss = model(**batch)
            if step == 0:
                print(loss.size())
        print("| ok is ", ok)



if __name__ == "__main__":
    main()














