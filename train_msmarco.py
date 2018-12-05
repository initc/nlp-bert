from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random
from tqdm import tqdm, trange

import numpy as np
from torch.nn import functional as F
import torch.nn as nn
import torch
import config

# import pdb
from sklearn.metrics import accuracy_score
from pytorch_pretrained_bert.qa_modeling import MSmarco
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.data.datasets import make_msmarco
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.data.bert_dataset import MSmarco_iterator



def main(args):


    logging = config.get_logging(args.log_name)
    logging.info("##"*20)
    logging.info("##"*20)
    logging.info("##"*20)
    logging.info(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.info("| question first :: {}".format(args.question_first))
    logging.info("| gpu count : {}".format(n_gpu))
    logging.info("| train batch size in each gpu : {}".format(args.train_batch_size))
    logging.info("| biuid tokenizer and model in : {}".format(args.pre_dir))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    tokenizer = BertTokenizer.build_tokenizer(args)
    train_data_iter = MSmarco_iterator(args, tokenizer, batch_size=args.train_batch_size, world_size=n_gpu, accumulation_steps=args.gradient_accumulation_steps, name="msmarco_train.pk")
    dev_data_iter = MSmarco_iterator(args, tokenizer, batch_size=args.train_batch_size, world_size=n_gpu, name="msmarco_dev.pk")
    data_size = len(train_data_iter)
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_train_steps = args.num_train_epochs*data_size//gradient_accumulation_steps
    # logging.info("| load dataset {}".format(data_size))
    logging.info("| train data size {}".format(len(train_data_iter)*n_gpu*args.train_batch_size))
    logging.info("| dev data size {}".format(len(dev_data_iter)*n_gpu*args.train_batch_size))
    logging.info("| train batch data size {}".format(len(train_data_iter)))
    logging.info("| dev batch data size {}".format(len(dev_data_iter)))
    logging.info("| total update {}".format(num_train_steps))

    model = MSmarco.build_model(args)
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
            if n_gpu==1:
                for key in batch.keys():
                    batch[key]=batch[key].to(device)
            loss = model(**batch)
            # pdb.set_trace()
            if n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss/args.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_update += 1
                if global_update % args.validate_updates==0:
                    validation(args, model, dev_data_iter, n_gpu, epochs, global_update, logging)
            if (step+1) % args.loss_interval==0:
                logging.info("TRAIN ::Epoch {} updates {}, train loss {}".format(epochs, global_update, loss.item()))
        # save_checkpoint(args, model, epochs)
        validation(args, model, dev_data_iter, n_gpu, epochs, global_update, logging)

def validation(args, model, data_iter, n_gpu, epochs, global_update, logging):

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
                logging.info("DEV :: epoch {} updates {}, valid loss {}".format(epochs, step, valid_loss))
                #print("DEV :: epoch {} updates {}, valid loss {}".format(epochs, step, valid_loss))
    score = total_hit_one/total_answer_n
    save_checkpoint(args, model, epochs, global_update, score, logging)
    logging.info("\n| Evaluation epoch {} updates {} : hit_one {}, hit_two {}, hit_three {}, accuracy_score {}".format(epochs, global_update, total_hit_one/total_answer_n, total_hit_two/total_answer_n, total_hit_three/total_answer_n, total_scores/data_lens))
    #print("\n| Evaluation epoch {} updates {} : hit_one {}, hit_two {}, hit_three {}, accuracy_score {}".format(epochs, global_update, total_hit_one/total_answer_n, total_hit_two/total_answer_n, total_hit_three/total_answer_n, total_scores/data_lens))
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

def save_checkpoint(args, model, epoch=0, updates=None. score=0, logging=None):
    best_scores = 0
    if hasattr(save_checkpoint, "best_scores"):
        best_scores = save_checkpoint.best_scores
    os.makedirs(args.save, exist_ok=True)
    if updates is not None:
        checkpoint_path = "checkpoints_{}_{}.pt".format(epoch, updates)
    else:
        checkpoint_path = "checkpoints_{}.pt".format(epoch)
    checkpoint_path = os.path.join(args.save, checkpoint_path)
    torch.save(model.state_dict(), checkpoint_path)
    ## save best 
    if score > best_scores:
        logging.info("save best checkpoint ::  epoch {} updates {}".format(epoch, updates))
        checkpoint_path = os.path.join(args.save, "checkpoints_best.pt")
        torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    args = config.parse_args()
    main(args)














