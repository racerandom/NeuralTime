#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
from collections import defaultdict
import argparse
from statistics import mean
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from sklearn.metrics import accuracy_score

import logging
from model import MultiTaskRelationClassifier
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

# def str2bool(v):
#     if isinstance(v, bool):
#        return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Multi-task Bert-based Temporal Relation Classifier')

parser.add_argument("-t", "--task", dest="task", default='ALL', type=str,
                    help="DCT, T2E, E2E, MAT or ALL")
parser.add_argument("-l", "--lab", dest="lab_type", default='6c', type=str,
                    help="lab_type, i.g. 4c, 6c or None")
parser.add_argument("-c", "--comp", dest="comp",
                    action='store_true',
                    help="complete match, True or False")
parser.add_argument("-b", "--batch", dest="BATCH_SIZE", default=16, type=int,
                    help="BATCH SIZE")
parser.add_argument("-e", "--epoch", dest="NUM_EPOCHS", default=7, type=int,
                    help="fine-tuning epoch number")
parser.add_argument("-o", "--order", dest="order",
                    action='store_true',
                    help="ordered training data by tasks")
parser.add_argument('--fp16',
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")

args = parser.parse_args()

logger.info('[args] Task: %s, label type: %s, complete agree: %s, ordered training:%s, batch_size: %i, epoch num: %i, fp16: %s' % (
    args.task,
    args.lab_type,
    str(args.comp),
    str(args.order),
    args.BATCH_SIZE,
    args.NUM_EPOCHS,
    str(args.fp16)
))

data_dir = 'data/merge/BCCWJ-TIMEX'


task_list = []
if args.task == 'ALL':
    task_list = ['DCT', 'T2E', 'E2E', 'MAT']
elif args.task == 'REV':
    task_list = ['MAT', 'E2E', 'T2E', 'DCT']
else:
    task_list = [args.task]

data_splits = preprocess.doc_kfold(data_dir)

cv_acc = defaultdict(lambda: [])

lab2ix = preprocess.get_lab2ix_from_type(args.lab_type)

logger.info(str(lab2ix))

eval_dict = defaultdict(lambda: defaultdict(lambda: []))

for cv_id, (train_files, test_files) in enumerate(data_splits):

    logger.info('[Cross Validation %i] train files %i, test files %i .' % (cv_id, len(train_files), len(test_files)))

    full_data_dict = defaultdict(lambda: {})

    for task in task_list:

        train_features, train_labs = preprocess.batch_make_tlink_instances_v2(
            train_files,
            task=task,
            lab_type=args.lab_type,
            comp=args.comp
        )

        test_features, test_labs = preprocess.batch_make_tlink_instances_v2(
            test_files,
            task=task,
            lab_type=args.lab_type,
            comp=args.comp
        )

        labs = train_labs + test_labs

        # lab2ix = utils.get_label2ix(labs)

        lab2count, major_lab, major_ratio = preprocess.count_major(labs)

        logger.info(str(lab2count))
        logger.info('major label: %s (%.2f%%) ' % (major_lab, major_ratio))
        full_data_dict[task]['lab2ix'] = lab2ix

        train_tensors = preprocess.instance_to_tensors(
            *train_features,
            train_labs,
            lab2ix
        )

        test_tensors = preprocess.instance_to_tensors(
            *test_features,
            test_labs,
            lab2ix
        )

        full_data_dict[task]['train_dataloader'] = DataLoader(
            train_tensors,
            batch_size=args.BATCH_SIZE,
            shuffle=True
        )
        full_data_dict[task]['test_dataloader'] = DataLoader(
            test_tensors,
            batch_size=args.BATCH_SIZE,
            shuffle=False
        )

        logger.info('Task %s, Train batch num: %i, Test batch num: %i' % (
            task,
            len(full_data_dict[task]['train_dataloader']),
            len(full_data_dict[task]['test_dataloader'])
        ))

    train_batch_seq, test_batch_seq = [], []

    for task in task_list:
        train_dataloader = full_data_dict[task]['train_dataloader']
        test_dataloader = full_data_dict[task]['test_dataloader']
        train_batch_seq += [task] * len(train_dataloader)
        test_batch_seq += [task] * len(test_dataloader)
        logger.info("Task %s, train batch %i, test batch %i" % (task, len(train_dataloader), len(test_dataloader)))

    if not args.order:
        import random
        random.shuffle(train_batch_seq)
    logger.info(str(train_batch_seq[:10]))

    num_train_optimization_steps = args.NUM_EPOCHS * len(train_batch_seq)

    model = MultiTaskRelationClassifier.from_pretrained(BERT_URL, num_labels=6 if (args.lab_type == '6c') else 4)
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0,
                              )
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)

        warmup_linear = WarmupLinearSchedule(warmup=0.1,
                                             t_total=num_train_optimization_steps)
    else:
        optimizer= BertAdam(optimizer_grouped_parameters,
                            lr=5e-5,
                            warmup=0.1,
                            t_total=num_train_optimization_steps)

    f1, acc = 0, 0
    global_step = 0
    model.train()
    for epoch in range(1, args.NUM_EPOCHS + 1):
        for task in task_list:
            full_data_dict[task]['iter_train_dataloader'] = iter(full_data_dict[task]['train_dataloader'])
            full_data_dict[task]['iter_test_dataloader'] = iter(full_data_dict[task]['test_dataloader'])

        for step, b_task in enumerate(tqdm(train_batch_seq)):

            b_tok, b_mask, b_sour_mask, b_targ_mask, b_sent_mask, b_lab = next(full_data_dict[b_task]['iter_train_dataloader'])
            loss = model(b_tok, b_sour_mask, b_targ_mask, b_task, token_type_ids=b_sent_mask, attention_mask=b_mask, labels=b_lab)

            if args.fp16:
                optimizer.backward(loss)
                # modify learning rate with special warm up BERT uses
                # if args.fp16 is False, BertAdam is used that handles this automatically
                lr_this_step = args.learning_rate * warmup_linear.get_lr(
                    global_step / num_train_optimization_steps,
                    0.1
                )
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
            else:
                loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            global_step += 1

    epoch_eval_dict = defaultdict(lambda: defaultdict(lambda: []))

    for b_task in test_batch_seq:
        b_tok, b_mask, b_sour_mask, b_targ_mask, b_sent_mask, b_lab = next(full_data_dict[b_task]['iter_test_dataloader'])
        b_pred_logits = model(b_tok, b_sour_mask, b_targ_mask, b_task, token_type_ids=b_sent_mask, attention_mask=b_mask)
        b_pred = torch.argmax(b_pred_logits, dim=-1).squeeze(1)
        epoch_eval_dict[b_task]['pred'] += b_pred.tolist()
        epoch_eval_dict[b_task]['gold'] += b_lab.tolist()

    for task in task_list:
        acc = accuracy_score(epoch_eval_dict[task]['gold'], epoch_eval_dict[task]['pred'])
        print('[Epoch %i] Task: %s, Accuracy: %.4f' % (args.NUM_EPOCHS, task, acc))

    for task in task_list:
        eval_dict[task]['pred'] += epoch_eval_dict[task]['pred']
        eval_dict[task]['gold'] += epoch_eval_dict[task]['gold']

for task in task_list:
    logger.info('Final Cross Validation Evaluation, Task: %s,  instance num: %i, ACC : %.4f' % (
        task,
        len(eval_dict[task]['gold']),
        accuracy_score(eval_dict[task]['gold'], eval_dict[task]['pred'])
    ))





