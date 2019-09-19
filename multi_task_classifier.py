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
from model import MultiTaskRelationClassifier, DocEmbMultiTaskTRC
import preprocess
import utils

from typing import Dict

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
n_gpu = torch.cuda.device_count()

if str(device) == 'cuda':
    PRETRAIN_BERT_DIR='/larch/share/bert/Japanese_models/Wikipedia/L-12_H-768_A-12_E-30_BPE'
else:
    PRETRAIN_BERT_DIR = '/Users/fei-c/Resources/embed/L-12_H-768_A-12_E-30_BPE'

logger = utils.init_logger('TLINK_Classifier', logging.INFO, logging.INFO)
logger.info('device: %s, gpu num: %i' % (device, n_gpu))


"""
main function
"""

parser = argparse.ArgumentParser(description='Multi-task Bert-based Temporal Relation Classifier')

parser.add_argument("-t", "--task", dest="task", default='ALL', type=str,
                    help="DCT, T2E, E2E, MAT or ALL")
parser.add_argument("-l", "--lab", dest="lab_type", default='6c', type=str,
                    help="lab_type, i.g. 4c, 6c or None")
parser.add_argument("-b", "--batch", dest="BATCH_SIZE", default=16, type=int,
                    help="BATCH SIZE")
parser.add_argument("-e", "--epoch", dest="NUM_EPOCHS", default=7, type=int,
                    help="fine-tuning epoch number")
parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")
parser.add_argument("--model_dir", default='checkpoints', type=str,
                    help="saved model dir for evaluation")
parser.add_argument("--comp",
                    action='store_true',
                    help="complete match, True or False")
parser.add_argument("--ordered",
                    action='store_true',
                    help="ordered training data by tasks")
parser.add_argument('--multi_gpu',
                    action='store_true',
                    help="wheter to use multiple gpus")
parser.add_argument('--seed',
                    type=int,
                    default=1029,
                    help="random seed for initialization")
parser.add_argument("--cache_dir",
                    default="cache",
                    type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")

args = parser.parse_args()

logger.info("""[args] Task: %s, label type: %s, do_train: %s, complete agree: %s, ordered training:%s, 
                      batch_size: %i, epoch num: %i, multi-gpu:%s""" % (
    args.task,
    args.lab_type,
    str(args.do_train),
    str(args.comp),
    str(args.ordered),
    args.BATCH_SIZE,
    args.NUM_EPOCHS,
    str(args.multi_gpu)
))

random.seed(1029)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

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

NUM_LABEL = len(lab2ix)

timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')

logger.info(str(lab2ix))

logger.info('Pretrained BERT dir: %s ' % PRETRAIN_BERT_DIR)

eval_dict = defaultdict(lambda: defaultdict(lambda: np.empty((0), int)))


for cv_id, (train_files, test_files) in enumerate(data_splits):

    logger.info('[Cross Validation %i] train files %i, test files %i .' % (cv_id, len(train_files), len(test_files)))

    CV_MODEL_DIR = '%s/%s_%s/cv%i' % (
        args.model_dir,
        args.task,
        timestamp_str,
        cv_id
    )

    output_model_file = os.path.join(CV_MODEL_DIR, WEIGHTS_NAME)
    output_config_file = os.path.join(CV_MODEL_DIR, CONFIG_NAME)

    if args.do_train:

        tokenizer = BertTokenizer.from_pretrained(PRETRAIN_BERT_DIR, do_lower_case=False, do_basic_tokenize=False)

        train_dataloader = {}

        train_batch_seq = []

        train_mid2ix = {}

        for task in task_list:

            train_features, train_labs = preprocess.batch_make_tlink_instances_v2(
                train_files,
                tokenizer,
                task=task,
                lab_type=args.lab_type,
                comp=args.comp
            )

            # lab2ix = utils.get_label2ix(labs)

            """ train labels distribution """
            train_lab2count, train_major_lab, train_major_ratio = preprocess.count_major(train_labs)

            logger.info(str(train_lab2count))
            logger.info('Train major label: %s (%.2f%%) ' % (train_major_lab, train_major_ratio))
            # full_data_dict[task]['lab2ix'] = lab2ix

            train_sour_mids, train_targ_mids = train_features[4], train_features[5]

            logger.debug(str(train_sour_mids[:5]))
            logger.debug(str(train_targ_mids[:5]))

            for mid in train_sour_mids:
                if mid not in train_mid2ix:
                    train_mid2ix[mid] = len(train_mid2ix)

            for mid in train_targ_mids:
                if mid not in train_mid2ix:
                    train_mid2ix[mid] = len(train_mid2ix)

            train_tensors = preprocess.instance_to_tensors(
                *train_features,
                train_labs,
                tokenizer,
                train_mid2ix,
                lab2ix
            )

            train_dataloader[task] = DataLoader(
                train_tensors,
                batch_size=args.BATCH_SIZE,
                shuffle=True
            )

            logger.info('Task %s, Train batch num: %i' % (
                task,
                len(train_dataloader[task])
            ))

            train_batch_num = len(train_dataloader[task])
            train_batch_seq += [task] * train_batch_num

        logger.info("Train mention ids num: %i" % len(train_mid2ix))

        if not args.ordered:
            random.shuffle(train_batch_seq)
        logger.info("Total batch num: %i, first 10 batch example: %s" % (
            len(train_batch_seq),
            str(train_batch_seq[:10])
        ))

        num_train_optimization_steps = args.NUM_EPOCHS * len(train_batch_seq)

        """ training """
        """ model initialization """
        model = DocEmbMultiTaskTRC.from_pretrained(
            PRETRAIN_BERT_DIR,
            num_emb=len(train_mid2ix),
            num_labels=NUM_LABEL,
            task_list=task_list
        )

        # model = nn.DataParallel(model)
        model.to(device)

        """ optimizer initialization """
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=5e-5,
                             warmup=0.1,
                             t_total=num_train_optimization_steps)

        """ epoch loops for training """
        global_step = 0
        model.train()
        for epoch in range(1, args.NUM_EPOCHS + 1):
            """ build a iterator of the dataloader, pop one batch every time """
            train_dataloader_iterator = {task: iter(train_dataloader[task]) for task in task_list}

            """ train steps """
            for step, b_task in enumerate(tqdm(train_batch_seq, desc="Training")):
                optimizer.zero_grad()

                batch = next(train_dataloader_iterator[b_task])

                b_tok, b_mask, b_sour_mask, b_targ_mask, b_sour_mid, b_targ_mid, b_sent_mask, b_lab = tuple(
                    t.to(device) for t in batch)

                loss = model(b_tok, b_sour_mask, b_targ_mask, b_task, token_type_ids=b_sent_mask, attention_mask=b_mask,
                             labels=b_lab)

                loss.backward()
                optimizer.step()
                global_step += 1

        if os.path.exists(CV_MODEL_DIR):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(CV_MODEL_DIR))
        if not os.path.exists(CV_MODEL_DIR):
            os.makedirs(CV_MODEL_DIR)

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(CV_MODEL_DIR)

        # torch.save(model, os.path.join(CV_MODEL_DIR, 'model.bin'))


    # else:
    #     """ eval mode
    #     reload the saved model
    #     """
    #     tokenizer = BertTokenizer.from_pretrained(CV_MODEL_DIR, do_lower_case=False, do_basic_tokenize=False)
    #     model = MultiTaskRelationClassifier.from_pretrained(CV_MODEL_DIR, num_labels=NUM_LABEL)
    #     model.to(device)

    """ Load the saved cv model """
    tokenizer = BertTokenizer.from_pretrained(CV_MODEL_DIR, do_lower_case=False, do_basic_tokenize=False)

    """ Evaluation at NUM_EPOCHS"""
    cv_eval_dict = defaultdict(lambda: defaultdict(lambda: np.empty((0), int)))

    """ Prepare test data """
    test_dataloader = {}

    test_batch_seq = []

    test_mid2ix = {}

    for task in task_list:

        test_features, test_labs = preprocess.batch_make_tlink_instances_v2(
            test_files,
            tokenizer,
            task=task,
            lab_type=args.lab_type,
            comp=args.comp
        )

        """ train labels distribution """
        test_lab2count, test_major_lab, test_major_ratio = preprocess.count_major(test_labs)

        logger.info(str(test_lab2count))
        logger.info('test major label: %s (%.2f%%) ' % (test_major_lab, test_major_ratio))
        # full_data_dict[task]['lab2ix'] = lab2ix

        test_sour_mids, test_targ_mids = test_features[4], test_features[5]

        logger.debug(str(test_sour_mids[:5]))
        logger.debug(str(test_targ_mids[:5]))

        for mid in test_sour_mids:
            if mid not in test_mid2ix:
                test_mid2ix[mid] = len(test_mid2ix)

        for mid in test_targ_mids:
            if mid not in test_mid2ix:
                test_mid2ix[mid] = len(test_mid2ix)

        test_tensors = preprocess.instance_to_tensors(
            *test_features,
            test_labs,
            tokenizer,
            test_mid2ix,
            lab2ix
        )

        test_dataloader[task] = DataLoader(
            test_tensors,
            batch_size=args.BATCH_SIZE,
            shuffle=False
        )

        test_batch_num = len(test_dataloader[task])
        test_batch_seq += [task] * test_batch_num

        logger.info("Task %s, test batch %i" % (
            task,
            test_batch_num
        ))

    logger.info("Test mention ids num: %i" % len(test_mid2ix))

    test_dataloader_iterator = {task: iter(test_dataloader[task]) for task in task_list}

    model = DocEmbMultiTaskTRC.from_pretrained(
        CV_MODEL_DIR,
        num_emb=len(train_mid2ix),
        num_labels=NUM_LABEL,
        task_list=task_list
    )
    model.load_state_dict(torch.load(output_model_file))
    # model = torch.load(os.path.join(CV_MODEL_DIR, 'model.bin'))
    model.to(device)

    """ Inference"""
    model.eval()
    for b_task in test_batch_seq:
        batch = next(test_dataloader_iterator[b_task])
        b_tok, b_mask, b_sour_mask, b_targ_mask, b_sour_mid, b_targ_mid, b_sent_mask, b_lab = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            b_pred_logits = model(b_tok, b_sour_mask, b_targ_mask, b_task, token_type_ids=b_sent_mask, attention_mask=b_mask, labels=None)
        b_pred = torch.argmax(b_pred_logits, dim=-1).squeeze(1)

        cv_eval_dict[b_task]['pred'] = np.append(cv_eval_dict[b_task]['pred'], b_pred.cpu().detach().numpy(), axis=0)
        cv_eval_dict[b_task]['gold'] = np.append(cv_eval_dict[b_task]['gold'], b_lab.cpu().detach().numpy(), axis=0)

    for task in task_list:
        acc = accuracy_score(cv_eval_dict[task]['gold'], cv_eval_dict[task]['pred'])
        logger.info('[Epoch %i] Task: %s, Accuracy: %.4f' % (args.NUM_EPOCHS, task, acc))
    logger.info('')

    for task in task_list:
        eval_dict[task]['pred'] = np.append(eval_dict[task]['pred'], cv_eval_dict[task]['pred'], axis=0)
        eval_dict[task]['gold'] = np.append(eval_dict[task]['gold'], cv_eval_dict[task]['gold'], axis=0)

for task in task_list:
    logger.info('Final Cross Validation Evaluation, Task: %s,  instance num: %i, ACC : %.4f' % (
        task,
        len(eval_dict[task]['gold']),
        accuracy_score(eval_dict[task]['gold'], eval_dict[task]['pred'])
    ))




