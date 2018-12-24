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
from pytorch_pretrained_bert.modeling import BertConfig
from pytorch_pretrained_bert.qa_modeling import MSmarco
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.data.bert_dataset import MSmarco_iterator

CONFIG_NAME = 'bert_config.json'

def main(args):

    logging = config.get_logging(args.log_name)
    logging.info(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    tokenizer = BertTokenizer.build_tokenizer(args)
    # train_data_iter = MSmarco_iterator(args, tokenizer, batch_size=args.train_batch_size, world_size=n_gpu, accumulation_steps=args.gradient_accumulation_steps, name="msmarco_train.pk")
    dev_data_iter = MSmarco_iterator(args, tokenizer, batch_size=args.valid_batch_size, world_size=n_gpu, name="msmarco_dev.pk")

    logging.info("| dev batch data size {}".format(len(dev_data_iter)))


    # num_train_steps = (96032//2//2)+(data_size-96032)//2
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    pre_dir = args.pre_dir
    config_file = os.path.join(pre_dir, CONFIG_NAME)
    bert_config = BertConfig.from_json_file(config_file)
    model = MSmarco(bert_config)
    logging.info("| load model from {}".format(args.path))
    state_dict = torch.load(args.path, map_location=torch.device('cpu'))
    metadata = getattr(state_dict, '_metadata', None)
    # state_dict = state_dict.copy()
    # if metadata is not None:
    #     state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='module.')

    if len(missing_keys) > 0:
        # logger.info("Weights of {} not initialized from pretrained model: {}".format(
        #     model.__class__.__name__, missing_keys))
        print("| Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        # logger.info("Weights from pretrained model not used in {}: {}".format(
            # model.__class__.__name__, unexpected_keys))
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))

    # model._load_from_state_dict(state_dict, prefix="module.")
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

        # save_checkpoint(args, model, epochs)
    validation(args, model, dev_data_iter, n_gpu, 0, 0, logging)

def merger_tensor(merge_batch):
    batch = {}
    keys = merge_batch[0].keys()
    for key in keys:
        batch[keys]=torch.cat([b[key] for b in merge_batch], dim = 0)
    return batch

def validation(args, model, data_iter, n_gpu, epochs, global_update, logging):

    total_hit_one = 0
    total_hit_two = 0
    total_hit_three = 0
    total_answer_n = 0
    total_hit_four = 0
    total_hit_five = 0
    total_scores = 0
    valid_loss = 0
    data_lens = 0
    criterion = nn.KLDivLoss()
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
            hit_one, hit_two, hit_three, hit_four, hit_five, answer_n = get_histest_score(targets, probs)
            total_hit_one += hit_one
            total_hit_two += hit_two
            total_hit_three += hit_three
            total_hit_four += hit_four
            total_hit_five += hit_five
            total_answer_n += answer_n
            if (step+1) % args.loss_interval==0:
                logging.info("DEV :: epoch {} updates {}, valid loss {}".format(epochs, step, valid_loss))
                #print("DEV :: epoch {} updates {}, valid loss {}".format(epochs, step, valid_loss))
    score = total_hit_one/total_answer_n
    # save_checkpoint(args, model, epochs, global_update, score, logging)
    logging.info("\n| Evaluation epoch {} updates {} : hit_one {}, hit_two {}, hit_three {}, hit_four {}, hit_five {}, accuracy_score {}".format(epochs, global_update, (total_hit_one/total_answer_n)*100, (total_hit_two/total_answer_n)*100, (total_hit_three/total_answer_n)*100, (total_hit_four/total_answer_n)*100, (total_hit_five/total_answer_n)*100, total_scores/data_lens))
    #print("\n| Evaluation epoch {} updates {} : hit_one {}, hit_two {}, hit_three {}, accuracy_score {}".format(epochs, global_update, total_hit_one/total_answer_n, total_hit_two/total_answer_n, total_hit_three/total_answer_n, total_scores/data_lens))
    #print("\n| Evaluation epoch {} updates {} : hit_one {}, hit_two {}, hit_three {}, accuracy_score {}".format(epochs, global_update, total_hit_one/total_answer_n, total_hit_two/total_answer_n, total_hit_three/total_answer_n, total_scores/data_lens), flush=True)

def get_histest_score(targets, probs):
    hit_one = 0
    hit_two = 0
    hit_three = 0
    hit_four = 0
    hit_five = 0
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
        if t[indic[0]]==1 or t[indic[1]]==1 or t[indic[2]]==1 or t[indic[3]]==1:
            hit_four += 1 
        if t[indic[0]]==1 or t[indic[1]]==1 or t[indic[2]]==1 or t[indic[3]]==1 or t[indic[4]]==1:
            hit_five += 1
    return hit_one, hit_two, hit_three, hit_four, hit_five, answer_n


if __name__ == "__main__":
    args = config.generate_args()
    main(args)














