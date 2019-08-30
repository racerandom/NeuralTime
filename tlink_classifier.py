#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
from argparse import ArgumentParser
from statistics import mean
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import logging
from model import RelationClassifier
import preprocess
import utils
import importlib

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

BERT_URL='/Users/fei-c/Resources/embed/L-12_H-768_A-12_E-30_BPE'
if str(device) == 'cuda':
    BERT_URL='/larch/share/bert/Japanese_models/Wikipedia/L-12_H-768_A-12_E-30_BPE'

logger = utils.init_logger('TLINK_Classifier', logging.INFO, logging.INFO)

"""
main function
"""
parser = ArgumentParser(description='Bert-based Temporal Relation Classifier')

parser.add_argument("-t", "--task", dest="task",
                    help="classification task, i.g. DCT, T2E, E2E and MAT")
parser.add_argument("-l", "--lab", dest="lab_type",
                    help="lab_type, i.g. 4c, 6c or None")
parser.add_argument("-c", "--comp", dest="comp", default=True, type=bool,
                    help="complete match, True or False")
parser.add_argument("-b", "--batch", dest="BATCH_SIZE", default=16,
                    help="BATCH SIZE")
parser.add_argument("-e", "--epoch", dest="NUM_EPOCHS", default=7,
                    help="fine-tuning epoch number")
args = parser.parse_args()

logger.info('[args] task: %s, label type: %s, complete agree: %s, batch_size: %i, epoch num: %i ' % (
    args.task, 
    args.lab_type,
    str(args.comp),
    args.BATCH_SIZE,
    args.NUM_EPOCHS
))

data_dir = 'data/merge/BCCWJ-TIMEX'

data_splits = preprocess.doc_kfold(data_dir)

pred_all, gold_all = [], []

for cv_id, (train_files, test_files) in enumerate(data_splits):

    logger.info('[Cross Validation %i] train files %i, test files %i .' % (cv_id, len(train_files), len(test_files)))

    tokenizer = BertTokenizer.from_pretrained(BERT_URL, do_lower_case=False, do_basic_tokenize=False)

    train_features, train_labs = preprocess.batch_make_tlink_instances_v2(
        train_files,
        tokenizer,
        task=args.task,
        lab_type=args.lab_type,
        comp=args.comp
    )

    test_features, test_labs = preprocess.batch_make_tlink_instances_v2(
        test_files,
        tokenizer,
        task=args.task,
        lab_type=args.lab_type,
        comp=args.comp
    )

    labs = train_labs + test_labs

    lab2ix = utils.get_label2ix(labs)

    lab2count, major_lab, major_ratio = preprocess.count_major(labs)

    logger.info(str(lab2count))
    logger.info('major label: %s (%.2f%%) ' % (major_lab, major_ratio))

    train_tensors = preprocess.instance_to_tensors(
        *train_features,
        train_labs,
        tokenizer,
        lab2ix,
        device
    )

    test_tensors = preprocess.instance_to_tensors(
        *test_features,
        test_labs,
        tokenizer,
        lab2ix,
        device
    )

    train_dataloader = DataLoader(train_tensors, batch_size=args.BATCH_SIZE,shuffle=True)
    test_dataloader = DataLoader(test_tensors, batch_size=args.BATCH_SIZE,shuffle=False)
    logger.info('Train batch num: %i, Test batch num: %i' % (len(train_dataloader), len(test_dataloader)))

    model = RelationClassifier.from_pretrained(BERT_URL, num_labels=len(lab2ix))
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerN®orm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer= BertAdam(optimizer_grouped_parameters,
                        lr=5e-5,
                        warmup=0.1,
                        t_total=args.NUM_EPOCHS * len(train_dataloader))

    f1, acc = 0, 0
    for epoch in range(1, args.NUM_EPOCHS + 1):
        epoch_loss = []
        for (b_tok, b_mask, b_sour_mask, b_targ_mask, b_sent_mask, b_lab) in tqdm(train_dataloader):
            model.train()
            model.zero_grad()
            loss = model(b_tok, b_sour_mask, b_targ_mask, token_type_ids=b_sent_mask, attention_mask=b_mask, labels=b_lab)

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        gold_cv, pred_cv = utils.pred_test(model, test_dataloader)
        p_cv, r_cv, f1_cv, _ = precision_recall_fscore_support(gold_cv, pred_cv, average='weighted')
        acc_cv = accuracy_score(gold_cv, pred_cv)
        print('[Epoch %i] Loss %.6f | Precision %.4f, Recall %.4f, F1 %.4f | Accuracy: %.4f' % (epoch, mean(epoch_loss), p_cv, r_cv, f1_cv, acc_cv))

    gold_all += (gold_cv)
    pred_all += (pred_cv)

p, r, f1, _ = precision_recall_fscore_support(gold_all, pred_all, average='weighted')
acc = accuracy_score(gold_all, pred_all)
logger.info('Final Cross Validation Evaluation: F1 : %.2f, ACC : %.2f%%' % (f1, acc))





