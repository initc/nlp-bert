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

from sklearn.metrics import accuracy_score
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
    parser.add_argument("--num-train-epochs", default=3, type=int,
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
    parser.add_argument('--threshold', type=int, default=0.36)
    parser.add_argument('--logfile', type=str, default="logfile.log")


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
    train_data_iter = MSmarco_iterator(args, tokenizer, batch_size=args.train_batch_size, world_size=n_gpu, name="msmarco_train.pk")
    dev_data_iter = MSmarco_iterator(args, tokenizer, batch_size=args.train_batch_size, world_size=n_gpu, name="msmarco_dev.pk")
    data_size = len(train_data_iter)
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

    global_update = 0
    for _ in range(args.num_train_epochs):

        for step, batch in enumerate(tqdm(train_data_iter, desc="Iteration")):
            model.train()
            loss = model(**batch)
            if n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            global_update += 1
            # if global_update % 10 == 0:
        validation(args, model, dev_data_iter, n_gpu)



def validation(args, model, data_iter, n_gpu):
    total_hit_one = 0
    total_hit_two = 0
    total_hit_three = 0
    total_answer_n = 0
    total_scores = 0

    for step, batch in enumerate(tqdm(data_iter, desc="Evaluation:")):
        model.eval()
        with torch.no_grad():
            targets = batch["targets"]
            batch["targets"] = None
            probs = model(**batch)
            # print("| prob is {}".format(probs))
            # print("| prob size {}".format(probs.size()))
            probs = probs.detach().cpu()
            targets = targets.numpy()
            probs = probs.numpy()
            b = probs.shape[0]
            total_scores = accuracy_score(targets, probs > args.threshold)*b
            hit_one, hit_two, hit_three, answer_n = get_histest_score(targets, probs)
            total_hit_one += hit_one
            total_hit_two += hit_two
            total_hit_three += hit_three
            total_answer_n += answer_n
    print("| Evaluation : hit_one {}, hit_two {}, hit_three {}, accuracy_score {}".format(total_hit_one/total_answer_n, total_hit_two/total_answer_n, total_hit_three/total_answer_n, total_scores/len(data_iter)))

def get_histest_score(targets, probs):
    hit_one = 0
    hit_two = 0
    hit_three = 0
    answer_n = 0
    for t,p in zip(targets, probs):
        if sum(t) == 0:
            continue
        answer_n += 1
        indic = np.argsort(-p)
        if t[indic[0]] == 1:
            hit_one += 1
        if t[indic[0]]==1 or t[indic[1]]==1:
            hit_two += 1
        if t[indic[0]]==1 or t[indic[1]]==1 or t[indic[2]]==1:
            hit_three += 1
    return hit_one, hit_two, hit_three, answer_n


if __name__ == "__main__":
    main()














