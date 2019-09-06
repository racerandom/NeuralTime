import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from pytorch_pretrained_bert import BertModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

class CertaintyClassifier(BertPreTrainedModel):
    
    def __init__(self, config, num_labels):
        super(CertaintyClassifier, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, dm_mask, token_type_ids=None, attention_mask=None, labels=None):
        last_layer_out, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        tag_rep = torch.bmm(dm_mask.unsqueeze(1).float(), last_layer_out)
        pooled_output = self.dropout(tag_rep)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        

class RelationClassifier(BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super(RelationClassifier, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, sour_mask, targ_mask, token_type_ids=None, attention_mask=None, labels=None):
        last_layer_out, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sour_rep = torch.bmm(sour_mask.unsqueeze(1).float(), last_layer_out)
        targ_rep = torch.bmm(targ_mask.unsqueeze(1).float(), last_layer_out)
        pooled_output = self.dropout(torch.cat((sour_rep, targ_rep), dim=-1))
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class MultiTaskRelationClassifier(BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super(MultiTaskRelationClassifier, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifiers = {task:nn.Linear(config.hidden_size * 2, num_labels).to(device) for task in ['DCT', 'T2E', 'E2E', 'MAT']}
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, sour_mask, targ_mask, task, token_type_ids=None, attention_mask=None, labels=None):
        last_layer_out, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sour_rep = torch.bmm(sour_mask.unsqueeze(1).float(), last_layer_out)
        targ_rep = torch.bmm(targ_mask.unsqueeze(1).float(), last_layer_out)
        pooled_output = self.dropout(torch.cat((sour_rep, targ_rep), dim=-1))
        logits = self.classifiers[task](pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class DocEmbMultiTaskTRC(BertPreTrainedModel):

    def __init__(self, config, num_emb, num_labels):
        super(DocEmbMultiTaskTRC, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.MentEmb = nn.Embedding(num_emb, config.hidden_size)
        self.classifiers = {task:nn.Linear(config.hidden_size * 2, num_labels).to(device) for task in ['DCT', 'T2E', 'E2E', 'MAT']}
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, sour_mask, targ_mask, task, token_type_ids=None, attention_mask=None, labels=None):
        last_layer_out, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sour_rep = torch.bmm(sour_mask.unsqueeze(1).float(), last_layer_out)
        targ_rep = torch.bmm(targ_mask.unsqueeze(1).float(), last_layer_out)
        pooled_output = self.dropout(torch.cat((sour_rep, targ_rep), dim=-1))
        logits = self.classifiers[task](pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

        
class SeqCertClassifier(BertPreTrainedModel):
    
    def __init__(self, config, num_labels):
        super(SeqCertClassifier, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, ner_masks, ner_clab_masks, token_type_ids=None, attention_mask=None, labels=None):
        last_layer_out, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        tag_rep = torch.bmm(ner_masks.float(), last_layer_out)
        pooled_output = self.dropout(tag_rep)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            active_loss = ner_clab_masks.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return logits