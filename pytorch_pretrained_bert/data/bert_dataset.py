import torch
import numpy as np
import pickle

from pytorch_pretrained_bert.data.datasets import get_batch

import os

class MSmarco_iterator():

    def __init__(self, args, tokenizer, batch_size=1, world_size=4, accumulation_steps=1, name="msmarco_train.pk"):

        self.pad_idx = tokenizer.pad()
        self.cls_idx = tokenizer.cls()
        self.sep_idx = tokenizer.sep()
        self.max_query_len = args.max_query_tokens
        self.max_passage_len = args.max_passage_tokens
        self.batch_size = batch_size
        self.world_size = world_size
        self.accumulation_steps = accumulation_steps
        self.dataset = pickle.load(open(os.path.join(args.data, name), "rb"))

    def __iter__(self):
        return get_batch(self.dataset, self.pad_idx, self.cls_idx, self.sep_idx, self.max_query_len, self.max_passage_len, self.batch_size, self.world_size)

    def __len__(self):
        return self.dataset["query_ids"].__len__()//self.world_size//self.batch_size