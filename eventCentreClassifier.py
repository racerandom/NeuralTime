#!/usr/bin/env python
# coding: utf-8
import os
from tqdm import tqdm
from collections import defaultdict
import argparse
import random
from datetime import datetime
from statistics import mean
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from pytorch_pretrained_bert import BertTokenizer, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from sklearn.metrics import accuracy_score
import torch.multiprocessing as mp

import logging
from model import SeqEventTRC
import preprocess
import utils
from collections import defaultdict

from typing import Dict

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
n_gpu = torch.cuda.device_count()
torch.manual_seed(1029)

if str(device) == 'cuda':
    PRETRAIN_BERT_DIR='/larch/share/bert/Japanese_models/Wikipedia/L-12_H-768_A-12_E-30_BPE'
else:
    PRETRAIN_BERT_DIR = '/Users/fei-c/Resources/embed/L-12_H-768_A-12_E-30_BPE'

tokenizer = BertTokenizer.from_pretrained(PRETRAIN_BERT_DIR, do_lower_case=False, do_basic_tokenize=False)

timebank_dir = 'data/merge/BCCWJ-TIMEX'
data_splits = preprocess.doc_kfold(timebank_dir)

reverse_6c = {
    'AFTER': 'BEFORE',
    'BEFORE': 'AFTER',
    'OVERLAP-OR-AFTER': 'BEFORE-OR-OVERLAP',
    'BEFORE-OR-OVERLAP': 'OVERLAP-OR-AFTER',
    'OVERLAP': 'OVERLAP',
    'VAGUE': 'VAGUE'
}


def detect_order(file_list):
    """ detect whether source mentions are always located before target mentions in documents """
    for single_file in file_list:
        event_centre_tlinks = {}
        event_counter = defaultdict(lambda: 0)

        doc_deunk_toks, doc_toks, doc_mid2smask, doc_tlinks = preprocess.extract_sents_from_xml_v2(
            single_file,
            tokenizer,
            lab_type='6c',
            comp=True
        )

        for k, v in doc_tlinks.items():
            for s, t, l in v:
                if k in ['DCT', 'T2E']:
                    event_counter[t] += 1
                else:
                    event_counter[s] += 1
                    s_sid, s_mask = doc_mid2smask[s]
                    t_sid, t_mask = doc_mid2smask[t]

                    if t_sid < s_sid:
                        raise Exception("t before s")
                    elif t_sid == s_sid:
                        if t_mask.index(1) < s_mask.index(1):
                            raise Exception("t before s")

        print('%.4f' % mean(event_counter.values()))


def prepare_doc_feats(file_list):
    """detect whether source mentions are always located before target mentions in documents"""

    e_toks, e_masks, events, labs, tasks = [], [], [], [], []
    for single_file in file_list:
        # print(single_file)
        """[[e_1, t_0, e_2], ['overlap', 'before', 'overlap'], ['E2E', 'DCT', 'E2E']]"""
        event_centre_data = defaultdict(lambda: [[], [], []])
        doc_deunk_toks, doc_toks, doc_mid2smask, doc_tlinks = preprocess.extract_sents_from_xml_v2(
            single_file,
            tokenizer,
            lab_type='6c',
            comp=True
        )

        doc_mid2smask['t0'] = [0, [0] * len(doc_toks[0])]

        for task in ['DCT', 'T2E', 'E2E', 'MAT']:
            for s, t, l in doc_tlinks[task]:
                if task in ['DCT', 'T2E']:
                    if s in doc_mid2smask:
                        if t not in event_centre_data:
                            event_centre_data[t][0].append(t)
                            event_centre_data[t][1].append('OVERLAP')
                            event_centre_data[t][2].append('E2E')
                        event_centre_data[t][0].append(s)
                        event_centre_data[t][1].append(reverse_6c[l])
                        event_centre_data[t][2].append(task)
                elif task in ['E2E', 'MAT']:
                    if s not in event_centre_data:
                        event_centre_data[s][0].append(s)
                        event_centre_data[s][1].append('OVERLAP')
                        event_centre_data[s][2].append('E2E')
                    if t not in event_centre_data[s][0]:
                        event_centre_data[s][0].append(t)
                        event_centre_data[s][1].append(l)
                        event_centre_data[s][2].append(task)
        # print(event_centre_data['e1'])

        for k, (e, l, t) in event_centre_data.items():
            assert len(e) == len(l) == len(t)

        doc_events = [e for e, l, t in event_centre_data.values()]
        doc_labs = [l for e, l, t in event_centre_data.values()]
        doc_tasks = [t for e, l, t in event_centre_data.values()]
        doc_e_toks = [[doc_toks[doc_mid2smask[e][0]] for e in es] for es in doc_events]
        doc_e_masks = [[doc_mid2smask[e][1] for e in es] for es in doc_events]

        events += doc_events
        labs += doc_labs
        tasks += doc_tasks
        e_toks += doc_e_toks
        e_masks += doc_e_masks
    max_len = max([len(e_feat) for e in e_toks for e_list in e for e_feat in e_list])
    print(max_len)
    mean_event_num = mean([len(f) for f in events])
    min_event_num = min([len(f) for f in events])
    max_event_num = max([len(f) for f in events])

    print('Total event num: %i, max token length: %i' % (len(events), max_len))
    print('mean num of relevant tlinks to a given event: %.4f' %
          mean_event_num)
    print('min num of relevant tlinks to a given event: %.0f' %
          min_event_num)
    print('max num of relevant tlinks to a given event: %.0f' %
          max_event_num)
    return e_toks, e_masks, events, labs, tasks, max_len, max_event_num


train_toks, train_masks, train_events, train_labs, train_tasks, train_max_len, train_max_event_num = prepare_doc_feats(data_splits[0][0])
test_toks, test_masks, test_events, test_labs, test_tasks, test_max_len, test_max_event_num = prepare_doc_feats(data_splits[0][1])

print("Total Tlink NUM: %i" % len([l for e_l in (train_labs + test_labs) for l in e_l]))
max_len = max(train_max_len, test_max_len)
max_event_num = max(train_max_event_num, test_max_event_num)
lab2ix = utils.get_label2ix([l for e_l in (train_labs + test_labs) for l in e_l])

NUM_EPOCHS = 20
NUM_LABEL = len(lab2ix)
task_list = ['DCT', 'T2E', 'E2E', 'MAT']

def padding_event_centre_feats(tokenizer, toks, masks, labs, lab2ix, max_len, max_event_num=1):
    padded_t_l, padded_id_t, padded_sm_t, padded_m_t, padded_labs = [], [], [], [], []
    for i in range(len(toks)):
        padded_el_t, padded_el_id, padded_el_sm, padded_el_m = [], [], [], []
        for e_t, e_m in zip(toks[i], masks[i]):
            padded_e_t = utils.padding_1d(['[CLS]'] + e_t, max_len + 1, pad_tok='[PAD]')
            padded_e_id = tokenizer.convert_tokens_to_ids(padded_e_t)
            padded_e_sm = utils.padding_1d([1] * (len(e_t) + 1), max_len + 1, pad_tok=0)
            padded_e_m = utils.padding_1d([0] + e_m, max_len + 1, pad_tok=0)
            assert len(padded_e_t) == len(padded_e_sm) == len(padded_e_m)
            padded_el_t.append(padded_e_t)
            padded_el_id.append(padded_e_id)
            padded_el_sm.append(padded_e_sm)
            padded_el_m.append(padded_e_m)
        padded_t_l.append(padded_el_t)
        padded_id_t.append(torch.tensor(padded_el_id))
        padded_sm_t.append(torch.tensor(padded_el_sm))
        padded_m_t.append(torch.tensor(padded_el_m))
        padded_labs.append(torch.tensor([lab2ix[l] for l in labs[i]]))
    return padded_t_l, padded_id_t, padded_sm_t, padded_m_t, padded_labs

train_padded_tok, train_id_t, train_sm_t, train_m_t, train_padded_labs = padding_event_centre_feats(tokenizer, train_toks, train_masks, train_labs, lab2ix, 100)
test_padded_tok, test_id_t, test_sm_t, test_m_t, test_padded_labs = padding_event_centre_feats(tokenizer, test_toks, test_masks, test_labs, lab2ix, 100)

train_dataloader = (train_id_t, train_sm_t, train_m_t, train_tasks, train_padded_labs)
test_dataloader = (test_id_t, test_sm_t, test_m_t, test_tasks, test_padded_labs)

c = list(zip(train_id_t, train_sm_t, train_m_t, train_tasks, train_padded_labs))
random.seed(1029)
random.shuffle(c)
train_id_t, train_sm_t, train_m_t, train_tasks, train_padded_labs = zip(*c)

model = SeqEventTRC.from_pretrained(
            PRETRAIN_BERT_DIR,
            num_labels=NUM_LABEL,
            task_list=task_list
        )
model.to(device)

""" optimizer initialization """
param_optimizer = list(model.named_parameters())
customized_layers = ['ment_embedding.weight']
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in customized_layers)],
     'lr': 5e-5,
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay + customized_layers)],
     'lr':5e-5,
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'lr':5e-5,
     'weight_decay': 0.0}
]
optimizer = BertAdam(optimizer_grouped_parameters,
                     warmup=0.1,
                     t_total=NUM_EPOCHS * len(train_padded_tok))

UPDATE_STEP = 5
accumulated_loss = 0.


for epoch in range(1, NUM_EPOCHS + 1):
    epoch_loss = []
    step = 1
    model.train()
    """ build a iterator of the dataloader, pop one batch every time """
    for b_toks, b_sm, b_em, b_tasks, b_labs in zip(train_id_t, train_sm_t, train_m_t, train_tasks, train_padded_labs):
        
        if epoch > 3:
            if epoch <15:
                for param in model.bert.parameters():
                    param.requires_grad = False
            else:
                for param in model.bert.parameters():
                    param.requires_grad = True

        loss = model(b_toks.to(device), b_em.to(device), b_tasks, attention_mask=b_sm.to(device), labels=b_labs.to(device))
        epoch_loss.append(loss.item())
        if step % 400 == 0 or step == len(train_padded_labs):
            print("Epoch %i, step %i, loss: %.6f" % (epoch, step, mean(epoch_loss)))
         
#        loss = loss / UPDATE_STEP
#        loss.backward()
        accumulated_loss += loss    

        """ accumulated updating """
        if step % UPDATE_STEP == 0 or step == len(train_padded_labs):
#            if random.random() < 0.5:
#                for param in model.bert.parameters():
#                    param.requires_grad = False
#            else:
#                for param in model.bert.parameters():
#                    param.requires_grad = True

            accumulated_loss = accumulated_loss / UPDATE_STEP
            accumulated_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            accumulated_loss = 0.

        step += 1

    """ Evaluate each epoch """
    corr_num = defaultdict(lambda: 0)
    total_num = defaultdict(lambda: 0)

    model.eval()
    with torch.no_grad():
        for b_toks, b_sm, b_em, b_tasks, b_labs in zip(test_id_t, test_sm_t, test_m_t, test_tasks, test_padded_labs):
            logits = model(b_toks.to(device), b_em.to(device), b_tasks, attention_mask=b_sm.to(device))
            for p, g, c in zip(logits.argmax(-1), b_labs.to(device)[1:], b_tasks[1:]):
                total_num[c] += 1
                if p == g:
                    corr_num[c] += 1

        for k in total_num.keys():
            print('%s acc: %.4f' % (k, corr_num[k] / total_num[k]))



