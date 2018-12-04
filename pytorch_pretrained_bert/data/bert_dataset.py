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
        self.question_first = args.question_first
        self.max_query_len = args.max_query_tokens
        self.max_passage_len = args.max_passage_tokens
        self.batch_size = batch_size
        self.world_size = world_size
        self.accumulation_steps = accumulation_steps
        self.dataset = pickle.load(open(os.path.join(args.data, name), "rb"))
        self.truncation(self.max_query_len, self.max_passage_len)
        self.sort_by_length()
        self.shuffle()

    def shuffle(self, portion=0.15):
        queries = self.dataset["query_ids"]
        data_lens = len(queries)
        save_count = int(data_lens*portion)
        passages = []
        targets = list(np.array(self.dataset["targets"])[:save_count])
        for i in range(10):
            passages.append(list(np.array(self.dataset["passage_ids"][i])[:save_count]))
        queries=list(np.array(queries)[:save_count])
        self.dataset["query_ids"]=queries
        self.dataset["passage_ids"] = passages
        self.dataset["targets"] = targets

    def truncation(self, max_query_len, max_passage_len):
        queries = self.dataset["query_ids"]
        passages = self.dataset["passage_ids"]
        b_size = len(queries)
        # query_truncated = 0
        # passage_truncated = 0
        for idx in range(b_size):
            if len(queries[idx]) > max_query_len:
                # query_truncated += 1
                queries[idx] = queries[idx][:max_query_len]
        for idx in range(len(passages)):
            for idy in range(b_size):
                if len(passages[idx][idy]) > max_passage_len:
                    # passage_truncated += 1
                    passages[idx][idy] = passages[idx][idy][:max_passage_len]

    def sort_by_length(self):
        queries = self.dataset["query_ids"]
        passages = self.dataset["passage_ids"]

        num_exs = len(queries)
        lengths = []
        for i in range(num_exs):
            lengths.append(len(queries[i]) + max([len(passages[j][i]) for j in range(10)]))
        lengths = np.array(lengths)
        indices = np.argsort(lengths)

        # just for test
        # indices = indices[:indices.size//10]
        self.dataset["query_ids"] = list(np.array(queries)[indices])
        for i in range(10):
            self.dataset["passage_ids"][i] = list(np.array(passages[i])[indices])
        self.dataset["targets"] = list(np.array(self.dataset["targets"])[indices])

    def __iter__(self):
        # print("| dataset iter")
        return get_batch(self.dataset, self.pad_idx, self.cls_idx, self.sep_idx, self.batch_size, self.world_size, accumulation_steps=self.accumulation_steps, question_first=self.question_first)

    def __len__(self):
        return (self.dataset["query_ids"].__len__()//(self.world_size*self.batch_size*self.accumulation_steps))*self.accumulation_steps


    