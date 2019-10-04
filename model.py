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

    def __init__(self, config, num_emb, num_labels, task_list):
        super(DocEmbMultiTaskTRC, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ment_embedding = nn.Embedding(num_emb, config.hidden_size)
        self.common_lab_rep = nn.Parameter(torch.Tensor(num_labels, config.hidden_size * 2).uniform_())
        for task in task_list:
            setattr(self, '%s_mapper' % task, nn.Linear(config.hidden_size * 2, config.hidden_size * 2))
        for task in task_list:
            setattr(self, '%s_classifier' % task, nn.Linear(config.hidden_size * 2, num_labels))
            getattr(self, '%s_classifier' % task).weight.data = getattr(self, '%s_mapper' % task)(self.common_lab_rep)
            # print(getattr(self, '%s_classifier' % task).weight.data.shape)
        # for task in task_list:
        #     setattr(self, '%s_classifier' % task, nn.Linear(config.hidden_size * 2, num_labels))
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, sour_mask, targ_mask, task, token_type_ids=None, attention_mask=None, labels=None):
        batch_size, _ = input_ids.shape
        last_layer_out, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # import pdb; pdb.set_trace()
        if task != 'DCT':
            sour_rep = torch.bmm(sour_mask.unsqueeze(1).float(), last_layer_out)
        else:
            sour_rep = self.ment_embedding(torch.zeros([batch_size, 1], dtype=torch.long).cuda())
        targ_rep = torch.bmm(targ_mask.unsqueeze(1).float(), last_layer_out)
        pooled_output = self.dropout(torch.cat((sour_rep, targ_rep), dim=-1))
        logits = getattr(self, '%s_classifier' % task)(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class DocMultiTaskTRC(BertPreTrainedModel):

    def __init__(self, config, num_emb, num_labels, task_list):
        super(DocMultiTaskTRC, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ment_embedding = nn.Embedding(num_emb, config.hidden_size)
        self.common_lab_rep = nn.Parameter(torch.tensor(num_labels, config.hidden_size * 2))
        
#        for task in task_list:
#            setattr(self, '%s_classifier' % task, nn.Linear(config.hidden_size * 2, num_labels))
#        for task in task_list:
#            setattr(self, '%s_mapper' % task, nn.Linear(config.hidden_size * 2, config.hidden_size * 2))
#        for task in task_list:
#            setattr(self, '%s_classifier' % task, nn.Linear(config.hidden_size * 2, num_labels))
#            getattr(self, '%s_classifier' % task).weight = getattr(self, '%s_classifier' % task)(self.common_lab_rep)
#

        self.apply(self.init_bert_weights)


    def forward(self, input_ids, sour_mask, targ_mask, task, token_type_ids=None, attention_mask=None, labels=None):
        batch_size, _ = input_ids.shape
        last_layer_out, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # import pdb; pdb.set_trace()
        if task != 'DCT':
            sour_rep = torch.bmm(sour_mask.unsqueeze(1).float(), last_layer_out)
        else:
            sour_rep = self.ment_embedding(torch.zeros([batch_size, 1], dtype=torch.long).cuda())
        targ_rep = torch.bmm(targ_mask.unsqueeze(1).float(), last_layer_out)
        pooled_output = self.dropout(torch.cat((sour_rep, targ_rep), dim=-1))
        logits = getattr(self, '%s_classifier' % task)(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class SeqEventTRC(BertPreTrainedModel):

    def __init__(self, config, num_labels, task_list):
        super(SeqEventTRC, self).__init__(config)
        self.num_labels = num_labels
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config)
        self.event_rnn = nn.GRU(config.hidden_size, config.hidden_size, 2,  batch_first=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ment_embedding = nn.Embedding(1, config.hidden_size)
        for task in task_list:
            setattr(self, '%s_classifier' % task, nn.Linear(config.hidden_size * 2, num_labels))
        self.apply(self.init_bert_weights)

    def init_rnn_hidden(self, layer_num, batch_size):
        return torch.zeros(layer_num, batch_size, self.hidden_size, device=device)

    def forward(self, seq_tok_ids, seq_e_m, seq_e_task, token_type_ids=None, attention_mask=None, labels=None):
        # import pdb; pdb.set_trace()
        batch_size, _ = seq_tok_ids.shape
        last_layer_out, _ = self.bert(seq_tok_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        event_rnn_in = self.dropout(torch.bmm(seq_e_m.unsqueeze(1).float(), last_layer_out).transpose(0, 1))
        hidden = self.init_rnn_hidden(2, 1)
        event_rnn_out, _ = self.event_rnn(event_rnn_in, hidden)

        logits_list = []

        for i in range(1, len(seq_e_task)):
            pooled_output = torch.cat((event_rnn_in[:, i, :], self.dropout(event_rnn_out[:, i - 1, :])), dim=-1)
            logits_list.append(getattr(self, '%s_classifier' % seq_e_task[i])(pooled_output))

        logits = torch.cat(logits_list, dim=0)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)[1:])
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
