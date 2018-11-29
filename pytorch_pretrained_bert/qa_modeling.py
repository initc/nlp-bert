from .modeling import PreTrainedBertModel, BertModel
import torch.nn as nn
import  torch.nn.functional  as F

class MSmorco(PreTrainedBertModel):

    def __init__(self, config):
        super(MSmorco, self).__init__(config)

        self.bert = BertModel(config)
        self.p_clf = nn.Linear(config.hidden_size, 10)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, target_idx=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        clf_out = self.p_clf(sequence_output[:,0])

        if target_idx is not None:
            clf_logits = F.log_softmax(clf_out)
            probs = F.softmax(target_idx)
            criterion = nn.KLDivloss()
            loss = criterion(clf_logits, probs)
            return loss
        else:
            return F.softmax(clf_out)

    


