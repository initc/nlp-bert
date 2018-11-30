from .modeling import PreTrainedBertModel, BertModel
import torch.nn as nn
import  torch.nn.functional  as F

class MSmorco(PreTrainedBertModel):

    def __init__(self, config):
        super(MSmorco, self).__init__(config)

        self.bert = BertModel(config)
        self.p_clf = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, targets=None):
        # print("| device is {}, type is {}".format(input_ids.device, type(input_ids)))
        bsz, clfs, t= input_ids.size()
        input_ids = input_ids.view(-1, t)
        token_type_ids = token_type_ids.view(-1, t)
        attention_mask = attention_mask.view(-1, t)
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        clf_out = self.p_clf(sequence_output[:,0]).squeeze(-1)
        clf_out = clf_out.view(bsz, clfs)

        if self.training:
            clf_logits = F.log_softmax(clf_out, dim=-1)
            probs = F.softmax(targets, dim=-1)
            criterion = nn.KLDivLoss()
            loss = criterion(clf_logits, probs)
            return loss
        else:
            return F.softmax(clf_out)

    


