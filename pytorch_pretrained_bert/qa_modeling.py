from .modeling import PreTrainedBertModel, BertModel,BertOnlyNSPHead
import torch.nn as nn
import  torch.nn.functional  as F
import torch

class MSmarco(PreTrainedBertModel):

    def __init__(self, config):
        super(MSmarco, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)
        self.criterion = nn.KLDivLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, targets=None):
        bsz, clfs, t= input_ids.size()
        input_ids = input_ids.view(bsz*clfs, t)
        token_type_ids = token_type_ids.view(bsz*clfs, t)
        attention_mask = attention_mask.view(bsz*clfs, t)

        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        clf_score = logits.view(bsz, clfs)
        if self.training:
            loss = self.criterion(F.log_softmax(clf_score, 1), targets)
            return loss
        else:
            return clf_score

class ParallelMSmarco(PreTrainedBertModel):

    def __init__(self, config):
        super(ParallelMSmarco, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask):
        bsz, clfs, t= input_ids.size()
        input_ids = input_ids.view(bsz*clfs, t)
        token_type_ids = token_type_ids.view(bsz*clfs, t)
        attention_mask = attention_mask.view(bsz*clfs, t)

        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        clf_score = logits.view(bsz, clfs)
        if self.training:
            
            return F.log_softmax(clf_score, 1)
        else:
            return clf_score
    

# class MSmarco(PreTrainedBertModel):

#     def __init__(self, config):
#         super(MSmarco, self).__init__(config, num_labels=1)

#         self.bert = BertModel(config)
#         self.cls = BertOnlyNSPHead(config)
#         self.apply(self.init_bert_weights)

#     def forward(self, input_ids, token_type_ids, attention_mask, targets=None):
#         # print("| device is {}, type is {}".format(input_ids.device, type(input_ids)))
#         bsz, clfs, t= input_ids.size()
#         input_ids = input_ids.view(bsz*clfs, t)
#         token_type_ids = token_type_ids.view(bsz*clfs, t)
#         attention_mask = attention_mask.view(bsz*clfs, t)


#         _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         seq_relationship_score = self.cls(pooled_output)
#         seq_relationship_score = seq_relationship_score.view(bsz, clfs, 2)

#         clf_score = seq_relationship_score[:,:,0].contiguous()
#         if targets is not None:
#             criterion = nn.KLDivLoss()
#             cls_logits = F.log_softmax(clf_score, 1)
#             loss = criterion(cls_logits, targets)
#             return loss
#         else:
#             # clf_probs = F.softmax(clf_score, dim=-1)
#             return clf_score

