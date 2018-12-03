from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import random
from tqdm import tqdm, trange

import numpy as np
from torch.nn import functional as F
import torch.nn as nn
import torch
import logging

import pdb
from sklearn.metrics import accuracy_score
from pytorch_pretrained_bert.qa_modeling import MSmarco
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.data.datasets import make_msmarco
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.data.bert_dataset import MSmarco_iterator

logging.basicConfig(filename="msmarco_train_info.log", format = '%(asctime)s - %(levelname)s -  %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)


def main():
    parser = argparse.ArgumentParser()


    parser.add_argument("--save-dir", default="checkpoints", type=str,
                        help="path to save checkpoints")

    ## Other parameters
    parser.add_argument("--data", default="data", type=str, help="MSmarco train and dev data")
    parser.add_argument("--origin-data", default="data", type=str, help="MSmarco train and dev data, will be tokenizer")
    parser.add_argument("--path", default="data", type=str, help="path(s) to model file(s), colon separated")
    parser.add_argument("--save", default="checkpoints/MSmarco", type=str, help="path(s) to model file(s), colon separated")
    parser.add_argument("--pre-dir", type=str,
                        help="where the pretrained checkpoint")
    parser.add_argument("--log-name", type=str,
                        help="where logfile")
    parser.add_argument("--max-passage-tokens", default=200, type=int,
                        help="The maximum total input passage length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max-query-tokens", default=50, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument('--gradient-accumulation-steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train-batch-size", default=2, type=int, help="Total batch size for training.")
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
    parser.add_argument('--validate-updates', type=int, default=30000, metavar='N',
                       help='validate every N updates')
    parser.add_argument('--loss-interval', type=int, default=5000, metavar='N',
                       help='validate every N updates')
    args = parser.parse_args()
    # logging = logging.getlogging(args.log_name)

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
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_train_steps = args.num_train_epochs*data_size/gradient_accumulation_steps
    print("| load dataset {}".format(data_size))

    model = ParallelMSmarco.build_model(args)
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
    for epochs in range(args.num_train_epochs):
        total_loss = 0
        
        for step, batch in enumerate(tqdm(train_data_iter, desc="Train Iteration")):
            model.train()
            loss = model(**batch)
            # pdb.set_trace()
            if n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            global_update += 1
            if global_update % args.validate_updates==0:
                validation(args, model, dev_data_iter, n_gpu, epochs, global_update)
            if global_update % args.loss_interval==0:
                print("TRAIN ::Epoch {} updates {}, train loss {}".format(epochs, global_update, loss.item()))
        save_checkpoint(args, model, epochs)
        validation(args, model, dev_data_iter, n_gpu, epochs, global_update)

def validation(args, model, data_iter, n_gpu, epochs, global_update):
    total_hit_one = 0
    total_hit_two = 0
    total_hit_three = 0
    total_answer_n = 0
    total_scores = 0
    valid_loss = 0
    data_lens = 0
    criterion = nn.KLDivLoss()
    batch_size = args.train_batch_size * n_gpu
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(tqdm(data_iter, desc="Dev Iteration:")):
            targets = batch["targets"]
            batch["targets"] = None
            scores = model(**batch)
            scores = scores.detach().cpu()
            # pdb.set_trace()
            # print("| prob is {}".format(probs))
            # print("| prob size {}".format(probs.size()))
            probs = F.softmax(scores, 1)

            valid_loss = criterion(F.log_softmax(scores, dim=1), F.softmax(targets, dim=1)).item()

            targets = targets.numpy()
            probs = probs.numpy()
            b = probs.shape[0]
            data_lens += b
            total_scores += accuracy_score(targets, probs > args.threshold)*b
            hit_one, hit_two, hit_three, answer_n = get_histest_score(targets, probs)
            total_hit_one += hit_one
            total_hit_two += hit_two
            total_hit_three += hit_three
            total_answer_n += answer_n
            if (step+1) % args.loss_interval==0:
                print("DEV :: epoch {} updates {}, valid loss {}".format(epochs, step, valid_loss))
    print("\n| Evaluation epoch {} updates {} : hit_one {}, hit_two {}, hit_three {}, accuracy_score {}".format(epochs, global_update, total_hit_one/total_answer_n, total_hit_two/total_answer_n, total_hit_three/total_answer_n, total_scores/data_lens))
    #print("\n| Evaluation epoch {} updates {} : hit_one {}, hit_two {}, hit_three {}, accuracy_score {}".format(epochs, global_update, total_hit_one/total_answer_n, total_hit_two/total_answer_n, total_hit_three/total_answer_n, total_scores/data_lens), flush=True)

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

def save_checkpoint(args, model, epoch=0):
    checkpoint_path = "{}_epoch_{}.pt".format(args.save, epoch)
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main()














