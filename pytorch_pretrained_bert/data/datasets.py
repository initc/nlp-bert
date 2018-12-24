import os
import csv
import json
import numpy as np
import pickle
import torch
import pdb

from tqdm import tqdm

def msmarco_tokenizer(tokenizer, querys, passages, targets):
    query_ids = []
    for query in querys:
        tokens = tokenizer.tokenize(query.strip())
        query_ids.append(tokenizer.convert_tokens_to_ids(tokens))
    passage_ids = [[] for _ in range(len(passages))]
    for idx in range(len(passages)):
        for passage in passages[idx]:
            tokens = tokenizer.tokenize(passage)
            passage_ids[idx].append(tokenizer.convert_tokens_to_ids(tokens))
    return {"query_ids":query_ids, "passage_ids":passage_ids, "targets":targets}

    

def _msmarco(path):
    with open(path) as f:
        lines = f.readlines()[:1000]
        ps = [[] for _ in range(10)]
        qs = []
        ts = []
        for line in lines:
            passages, query, target = line.strip().split('\t')
            qs.append(query)
            ts.append(list(json.loads(target)))
            length = len(passages.split('@@'))
            for idx, passage in enumerate(passages.split('@@')):
                ps[idx].append(passage)
    return qs, ps, ts

def make_msmarco(args, tokenizer):
    train_queries, train_passages, train_targets = _msmarco(os.path.join(args.origin_data, 'train_processed_selector.txt'))
    train_data = msmarco_tokenizer(tokenizer, train_queries, train_passages, train_targets)
    dev_queries, dev_passages, dev_targets = _msmarco(os.path.join(args.origin_data, 'dev_processed_selector.txt'))
    dev_data = msmarco_tokenizer(tokenizer, dev_queries, dev_passages, dev_targets)
    with open(os.path.join(args.data, "msmarco_train.pk"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join(args.data, "msmarco_dev.pk"), "wb") as f:
        pickle.dump(dev_data, f)

def msmarco_fariseq(in_file, out_dir, pref="train"):
    train_queries, train_passages, train_targets = _msmarco(in_file)
    assert len(train_queries) == len(train_passages[0])
    assert len(train_passages) == 10
    assert len(train_targets) == len(train_queries)
    msmarco_A_file = os.path.join(out_dir, "{}-query.bert".format(pref))
    msmarco_B_file = os.path.join(out_dir, "{}-passage.bert".format(pref))
    msmarco_t_file = os.path.join(out_dir, "{}-target.pk".format(pref))
    with open(msmarco_A_file, "w") as f:
        for query in train_queries:
            for step in range(10):
                f.write("{}\n".format(query))
    with open(msmarco_B_file, "w") as f:
        for pa in zip(*train_passages):
            assert len(pa)==10
            for l in pa:
                f.write("{}\n".format(l))
    with open(msmarco_t_file, "wb") as f:
        pickle.dump(train_targets, f)





def _batch(data, pad_idx, cls_idx, sep_idx, batch_size, accumulation_steps=1, question_first=True):

    # def truncation(queries, passages):
    #     b_size = len(queries)
    #     query_truncated = 0
    #     passage_truncated = 0
    #     for idx in range(b_size):
    #         if len(queries[idx]) > max_query_len:
    #             query_truncated += 1
    #             queries[idx] = queries[idx][:max_query_len]
    #     for idx in range(len(passages)):
    #         for idy in range(b_size):
    #             if len(passages[idx][idy]) > max_passage_len:
    #                 passage_truncated += 1
    #                 passages[idx][idy] = passages[idx][idy][:max_passage_len]
    #     # print("| query truncated count {}".format(query_truncated))
    #     # print("| passage truncated count {}".format(passage_truncated))
    #     return queries, passages

    def max_len(queries, passages):
        num_exs = len(queries)
        lengths = []
        for i in range(num_exs):
            lengths.append(len(queries[i]) + max([len(passages[j][i]) for j in range(10)]))
        return max(lengths)

    # queries, passages = truncation(data["query_ids"], data["passage_ids"])
    # queries, passages = data["query_ids"], data["passage_ids"]

    # -----
    query_ids, passage_ids, targets = data["query_ids"], data["passage_ids"], data["targets"]
    # print("| in dataset {}".format(query_ids[0]))
    assert len(query_ids) == len(passage_ids[0])
    assert len(query_ids) == len(targets)
    batch_count = (len(query_ids)//(batch_size*accumulation_steps))*accumulation_steps
    for idx in range(batch_count):
        begin = idx*batch_size
        end = begin + batch_size
        # for save memory
        passage_for_len = []
        for i in range(len(passage_ids)):
            passage_for_len.append(passage_ids[i][begin:end])
        max_time_step = max_len(query_ids[begin:end], passage_for_len)
        # pdb.set_trace()
        query_tmp = np.array(query_ids[begin:end])
        passage_tmp = np.array(passage_for_len)

        input_tensor = np.zeros((batch_size, 10, max_time_step+3), dtype=np.int32)
        input_tensor.fill(pad_idx)
        input_type = np.zeros((batch_size, 10, max_time_step+3), dtype=np.int32)
        input_type.fill(pad_idx)
        input_mask = np.zeros((batch_size, 10, max_time_step+3), dtype=np.int32)
        input_mask.fill(0)

        input_tensor[:, :, 0] = cls_idx
        input_type[:, :, 0] = 0
        input_mask[:, :, 0] = 1

        for idx_batch in range(batch_size):
            query_len = len(query_tmp[idx_batch])
            assert len(passage_tmp)==10
            if question_first:
                input_tensor[idx_batch, :, 1:query_len+1] = query_tmp[idx_batch]
                input_type[idx_batch, :, 1:query_len+2] = 0
                input_mask[idx_batch, :, 1:query_len+2] = 1
                input_tensor[idx_batch, :, query_len+1] = sep_idx

                for idy in range(len(passage_tmp)):
                    passage_len = len(passage_tmp[idy][idx_batch])

                    input_tensor[idx_batch, idy, query_len+2:query_len+passage_len+2] = passage_tmp[idy, idx_batch]
                    input_type[idx_batch, idy, query_len+2:query_len+passage_len+3] = 1
                    input_mask[idx_batch, idy, query_len+2:query_len+passage_len+3] = 1
                    input_tensor[idx_batch, idy, query_len+passage_len+2] = sep_idx
            else:

                for idy in range(len(passage_tmp)):
                    passage_len = len(passage_tmp[idy][idx_batch])

                    input_tensor[idx_batch, idy, 1:passage_len+1] = passage_tmp[idy, idx_batch]
                    input_tensor[idx_batch, idy, passage_len+1] = sep_idx
                    input_type[idx_batch, idy, 1:passage_len+2] = 0
                    input_mask[idx_batch, idy, 1:passage_len+2] = 1
                    # questions 
                    input_tensor[idx_batch, idy, passage_len+2:query_len+passage_len+2] = query_tmp[idx_batch]
                    input_tensor[idx_batch, idy, query_len+passage_len+2] = sep_idx
                    input_type[idx_batch, idy, passage_len+2:query_len+passage_len+3] = 1
                    input_mask[idx_batch, idy, passage_len+2:query_len+passage_len+3] = 1

                
        yield {"input_ids":torch.LongTensor(input_tensor), "token_type_ids":torch.LongTensor(input_type), "attention_mask":torch.LongTensor(input_mask), "targets":torch.FloatTensor(targets[begin:end])}


def get_batch(data_dict, pad_idx, cls_idx, sep_idx, batch_size=1, world_size=4, accumulation_steps=1, question_first=True):

    for batch in _batch(data_dict, pad_idx, cls_idx, sep_idx, batch_size*world_size, accumulation_steps, question_first):
        yield batch




