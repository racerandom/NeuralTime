#!/usr/bin/env python
# coding: utf-8
import os
import mojimoji
from pyknp import Juman
import xml.etree.ElementTree as ET
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import TensorDataset
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import WEIGHTS_NAME, CONFIG_NAME

juman = Juman()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
if str(device) == 'cpu':
    BERT_URL='/Users/fei-c/Resources/embed/L-12_H-768_A-12_E-30_BPE'
elif str(device) == 'cuda':
    BERT_URL='/larch/share/bert/Japanese_models/Wikipedia/L-12_H-768_A-12_E-30_BPE'
tokenizer = BertTokenizer.from_pretrained(BERT_URL, do_lower_case=False, do_basic_tokenize=False)


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def get_label2ix(y_data):
    label2ix = {}
    for label in y_data:
        if label not in label2ix:
            label2ix[label] = len(label2ix)
    return label2ix


def padding_1d(seq_1d, max_len, pad_tok=0, direct='right'):
    for i in range(0, max_len - len(seq_1d)):
        if direct in ['right']:
            seq_1d.append(pad_tok)
        else:
            seq_1d.insert(0, pad_tok)
    return seq_1d

def padding_2d(seq_2d, max_len, pad_tok=0, direct='right'):

    for seq_1d in seq_2d:
        for i in range(0, max_len - len(seq_1d)):
            if direct in ['right']:
                seq_1d.append(pad_tok)
            else:
                seq_1d.insert(0, pad_tok)
    return seq_2d

def match_sbp_label(bpe_x, y):
    bpe_y = y.copy()
    for i in range(len(bpe_x)):
        if bpe_x[i].startswith('##'):
            if '-' in bpe_y[i-1]:
                bpe_y.insert(i, 'I' + bpe_y[i-1][1:])
            else:
                bpe_y.insert(i, bpe_y[i-1])
    assert len(bpe_x) == len(bpe_y)
    return bpe_y

def match_sbp_mask(bpe_x, y):
    bpe_y = y.copy()
    for i in range(len(bpe_x)):
        if bpe_x[i].startswith('##'):
            bpe_y.insert(i, bpe_y[i-1])
    assert len(bpe_x) == len(bpe_y)
    return bpe_y

def match_sbp_cert_labs(bpe_x, y):
    bpe_y = y.copy()
    for i in range(len(bpe_x)):
        if bpe_x[i].startswith('##'):
            bpe_y.insert(i, '_')
    return bpe_y


def explore_unk(bpe_x, ori_x):
    
    ix_count = 0
    deunk_bpe_x = []
    
    for tok in bpe_x:
        if not tok.startswith('##'):
            if tok != '[UNK]':
                deunk_bpe_x.append(tok)
            else:
                deunk_bpe_x.append(ori_x[ix_count])
            ix_count += 1
        else:
            deunk_bpe_x.append(tok)
    assert len(bpe_x)==len(deunk_bpe_x)
    return deunk_bpe_x

def write_data_to_txt(np_findings, file_name):
    with open(file_name, "w") as txt_file:
        for d in np_findings:
            if isinstance(d, str):
                txt_file.write(d + '\n')

def out_xml(orig_tok, pred_ix, ix2label):
    lines = []
    for sent_tok in orig_tok:
        label_prev = 'O'
        line_str = ''
        for tok in sent_tok:
            label = ix2label[pred_ix.pop()]
            if label_prev.startswith('O'):
                if label.startswith('B'):
                    line_str += '<%s>%s' % (label.split('-')[-1], tok)
                elif label.startswith('I'):
                    line_str += '<%s>%s' % (label.split('-')[-1], tok)
                else:
                    line_str += tok
            elif label_prev.startswith('B'):
                if label.startswith('B'):
                    line_str += '</%s><%s>%s' % (label_prev.split('-')[-1], label.split('-')[-1], tok)
                elif label.startswith('I'):
                    line_str += tok
                else:
                    line_str += '</%s>%s' % (label_prev.split('-')[-1], tok)
            elif label_prev.startswith('I'):
                if label.startswith('B'):
                    line_str += '</%s><%s>%s' % (label_prev.split('-')[-1], label.split('-')[-1], tok)
                elif label.startswith('I'):
                    line_str += tok
                else:
                    line_str += '</%s>%s' % (label_prev.split('-')[-1], tok)
            label_prev = label 
        lines.append(line_str)
        
    return lines


def convert_clinical_data_to_conll(clinical_file, fo,  sent_tag=True, skip_empty=False, defaut_cert='_', is_raw=False):
    x_data, y_data = [], []
    with open(clinical_file, 'r') as fi:
        for index, line in enumerate(fi):
            line = line.strip().replace('\n', '').replace('\u3000', mojimoji.han_to_zen('-'))
        
            if skip_empty:
                if '<' not in line: ## skip the lines without any tag inside.
                    continue
                    
                if line[0] == '・':
                    line = line[1:]

                if line in ['<CHEST: CT>',
                            '<CHEST>',
                            '<胸部CT>',
                            '<CHEST；CT>',
                            '<胸部単純CT>',
                            '<ABD US>']:
                    continue
            
            if sent_tag:
                line = '<sentence>' + line + '</sentence>'
            
            if not is_raw:
                st = ET.fromstring(line)
                toks, labs, cert_labs = [], [], []
                for item in st.iter():
                    if item.text is not None:
                        seg = juman.analysis(item.text)
                        toks += [mojimoji.han_to_zen(w.midasi) for w in seg.mrph_list()]
                        if item.tag in ['event', 'TIMEX3', 'd', 'a', 'f', 'c']:
                            tok_labs = ['I-%s' % (item.tag.capitalize())] * len(seg)
                            tok_labs[0] = 'B-%s' % (item.tag.capitalize())
                            labs += tok_labs
                            if item.tag == 'd' and 'certainty' in item.attrib:
                                tok_cert_labs = ['_'] * len(seg)
                                tok_cert_labs[0] = item.attrib['certainty']
                                cert_labs += tok_cert_labs
                            else:
                                cert_labs += ['_'] * len(seg)
                        else:
                            labs += ['O'] * len(seg)
                            cert_labs += ['_'] * len(seg)
                    if item.tail is not None:
                        seg_tail = juman.analysis(item.tail)
                        toks += [mojimoji.han_to_zen(w.midasi) for w in seg_tail.mrph_list()]
                        labs += ['O'] * len(seg_tail)
                        cert_labs += ['_'] * len(seg_tail)
                assert len(toks) == len(labs) == len(cert_labs)

                sbp_toks = tokenizer.tokenize(' '.join(toks))
                deunk_toks = explore_unk(sbp_toks, toks)
                sbp_labs = match_sbp_label(deunk_toks, labs)
                sbp_cert_labs = match_sbp_cert_labs(deunk_toks, cert_labs)

            else:
                seg = juman.analysis(line)
                toks = [mojimoji.han_to_zen(w.midasi) for w in seg.mrph_list()]
                sbp_toks = tokenizer.tokenize(' '.join(toks))
                deunk_toks = explore_unk(sbp_toks, toks)
                sbp_labs = ['O'] * len(sbp_toks)
                sbp_cert_labs = ['_'] * len(sbp_toks)
                
            assert len(sbp_toks) == len(deunk_toks) == len(sbp_labs)  == len(sbp_cert_labs)
                
            for d, t, l, cl in zip(deunk_toks, sbp_toks, sbp_labs, sbp_cert_labs):
                fo.write('%s\t%s\t%s\t%s\n' % (d, t, l, cl))
            fo.write('\n')
    return index + 1
                
            
def batch_convert_clinical_data_to_conll(data_dir, file_out, sent_tag=True, skip_empty=False, defaut_cert='_', is_raw=False):
    sent_count = []
    with open(file_out, 'w') as fo:
        for file in os.listdir(data_dir):
            if file.endswith(".sent"):
                dir_file = os.path.join(data_dir, file)
                sent_count.append(convert_clinical_data_to_conll(dir_file, fo, sent_tag=sent_tag, skip_empty=skip_empty, defaut_cert=defaut_cert, is_raw=is_raw))
    print(sum(sent_count))
    return sent_count
                
                
def read_conll(conll_file):
    deunks, toks, labs, cert_labs = [], [], [], []
    with open(conll_file) as fi:
        sent_deunks, sent_toks, sent_labs, sent_cert_labs = [], [], [], []
        for line in fi:
            line = line.strip()
            if not line:
                if sent_deunks:
                    deunks.append(sent_deunks)
                    toks.append(sent_toks)
                    labs.append(sent_labs)
                    cert_labs.append(sent_cert_labs)
                    sent_deunks, sent_toks, sent_labs, sent_cert_labs = [], [], [], []
                continue
            deunk, tok, lab, cert_lab = line.split()
            sent_deunks.append(deunk)
            sent_toks.append(tok)
            sent_labs.append(lab)
            sent_cert_labs.append(cert_lab)
    return deunks, toks, labs, cert_labs


def extract_ner_from_conll(conll_file, tokenizer, lab2ix, device):
    deunks, toks, labs, cert_labs = read_conll(conll_file)
    max_len = max([len(x) for x in toks])
    pad_tok_ids, pad_masks, pad_lab_ids = [], [], []
    for tok, lab in zip(toks, labs):
        pad_tok = padding_1d(['[CLS]'] + tok, max_len + 1, pad_tok='[PAD]')
        pad_tok_id = tokenizer.convert_tokens_to_ids(pad_tok)
        pad_mask = padding_1d([1] * (len(tok) + 1), max_len + 1, pad_tok=0)
        pad_lab = padding_1d(['O'] + lab, max_len + 1, pad_tok='O')
        pad_lab_id = [lab2ix[lab] for lab in pad_lab]
        assert len(pad_tok_id) == len(pad_mask) == len(pad_lab_id)
        pad_tok_ids.append(pad_tok_id)
        pad_masks.append(pad_mask)
        pad_lab_ids.append(pad_lab_id)
    pad_tok_ids_t = torch.tensor(pad_tok_ids).to(device)
    pad_masks_t = torch.tensor(pad_masks).to(device)
    pad_lab_ids_t = torch.tensor(pad_lab_ids).to(device)
    print('ner data size:',
          pad_tok_ids_t.shape,
          pad_masks_t.shape,
          pad_lab_ids_t.shape)
    return TensorDataset(pad_tok_ids_t,
                         pad_masks_t,
                         pad_lab_ids_t), deunks


def ner_labels_to_masks(ner_labels, max_ner_num, cert_labels=None):
    
    ner_masks = np.zeros((max_ner_num, len(ner_labels)), dtype=int)
    ner_cert_labels = ['[PAD]'] * max_ner_num
    
    if not cert_labels:
        ner_offset = 0; prev_label = 'O'
        for i, curr_label in enumerate(ner_labels):
            if i > 0:
                prev_label = ner_labels[i - 1]
            if curr_label in ['B-D']:
                if prev_label in ['B-D', 'I-D']:
                    ner_offset += 1
                ner_masks[ner_offset][i] = 1
            elif curr_label in ['I-D']:
                if prev_label in ['B-D', 'I-D']:
                    ner_masks[ner_offset][i] = 1
                else:
                    ner_masks[ner_offset][i] = 1
            else:
                if prev_label in ['B-D', 'I-D']:
                    ner_offset += 1
    else:
        ner_offset = 0; prev_label = 'O'; prev_cert_label = '[PAD]'
        for i, (curr_label, curr_cert_label) in enumerate(zip(ner_labels, cert_labels)):
            if i > 0:
                prev_label = ner_labels[i - 1]
                prev_cert_label = cert_labels[i - 1]
            if curr_label in ['B-D']:
                if prev_label in ['B-D', 'I-D']:
                    ner_offset += 1
                ner_masks[ner_offset][i] = 1
                ner_cert_labels[ner_offset] = curr_cert_label if curr_cert_label != '_' else '[PAD]'
            elif curr_label in ['I-D']:
                if prev_label in ['B-D', 'I-D']:
                    ner_masks[ner_offset][i] = 1
                else:
                    ner_masks[ner_offset][i] = 1
                    ner_cert_labels[ner_offset] = curr_cert_label
            else:
                if prev_label in ['B-D', 'I-D']:
                    ner_offset += 1
            
    cert_label_masks = padding_1d([1] * len(set(ner_masks.nonzero()[0])), max_ner_num, pad_tok=0)
    
    return ner_masks, cert_label_masks, ner_cert_labels


def extract_cert_from_conll(conll_file, tokenizer, cert_lab2ix, device, test_mode=False):
    deunks, toks, labs, clabs = read_conll(conll_file)
    
    max_len = max([len(x) for x in toks])
    max_ner_num = max([s_l.count('B-D') for s_l in labs])
    
    pad_tok_ids, pad_masks, pad_ner_masks, clab_masks, pad_cert_lab_ids = [], [], [], [], []
    for s_toks, s_labs, s_clabs in zip(toks, labs, clabs):
        pad_s_toks = padding_1d(['[CLS]'] + s_toks, max_len + 1, pad_tok='[PAD]')
        pad_s_tok_ids = tokenizer.convert_tokens_to_ids(pad_s_toks)
        pad_s_masks = padding_1d([1] * (len(s_toks) + 1), max_len + 1, pad_tok=0)
        
        if test_mode:
            s_ner_masks, s_clab_masks, s_ner_clabs = ner_labels_to_masks(s_labs, max_ner_num, None)
        else:
            s_ner_masks, s_clab_masks, s_ner_clabs = ner_labels_to_masks(s_labs, max_ner_num, s_clabs)
        
        pad_s_ner_masks = padding_2d(np.insert(s_ner_masks, 0, 0, axis=1).tolist(), max_len + 1, pad_tok=0)
        
        pad_tok_ids.append(pad_s_tok_ids)
        pad_masks.append(pad_s_masks)
        pad_ner_masks.append(pad_s_ner_masks)
        clab_masks.append(s_clab_masks)
        pad_cert_lab_ids.append([cert_lab2ix[clab] for clab in s_ner_clabs])
        
        
        assert len(pad_tok_ids) == len(pad_masks)
        assert len(pad_ner_masks) == len(clab_masks) == len(pad_cert_lab_ids)
    
    pad_tok_ids_t = torch.tensor(pad_tok_ids).to(device)
    pad_masks_t = torch.tensor(pad_masks).to(device)
    pad_ner_masks_t = torch.tensor(pad_ner_masks).to(device)
    clab_masks_t = torch.tensor(clab_masks).to(device)
    pad_clab_ids_t = torch.tensor(pad_cert_lab_ids).to(device)
    print('cert data size:', 
          pad_tok_ids_t.shape, 
          pad_masks_t.shape, 
          pad_ner_masks_t.shape, 
          clab_masks_t.shape,
          pad_clab_ids_t.shape)
    
    return TensorDataset(pad_tok_ids_t, 
                         pad_masks_t, 
                         pad_ner_masks_t, 
                         clab_masks_t, 
                         pad_clab_ids_t
                        ), deunks


def init_logger(logger_name, level, stream_level):
    import logging

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(stream_level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    if (logger.hasHandlers()):
        logger.handlers.clear()
    # add ch to logger
    logger.addHandler(ch)
    return logger


def save_checkpoint(model, checkpoint_dir):
    # Goal: Save a model, configuration and vocabulary that you have fine-tuned

    # If we have a distributed model, save only the encapsulated model
    # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
    if os.path.exists(checkpoint_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(checkpoint_dir))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model_to_save = model.module if hasattr(model, 'module') else model

    # If we save using the predefined names, we can load using `from_pretrained`
    checkpoint_model_file = os.path.join(checkpoint_dir, WEIGHTS_NAME)
    checkpoint_config_file = os.path.join(checkpoint_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), checkpoint_model_file)
    model_to_save.config.to_json_file(checkpoint_config_file)
    tokenizer.save_vocabulary(checkpoint_dir)

            
def eval_pid_seq(model, tokenizer, test_data, orig_token, label2ix, epoch):
    lines = []
    model.eval()
    with torch.no_grad():
        with open('output_ep%i.txt' % epoch, 'w') as fo:
            for (token, mask, gold), ti in zip(test_data, orig_token):
                pred_prob = model(token, attention_mask=mask)
                pred = torch.argmax(pred_prob, dim=-1)

                t_masked_ix = torch.masked_select(token[:,1:], mask[:,1:].byte())
                pred_masked_ix = torch.masked_select(pred[:,1:], mask[:,1:].byte())
                gold_masked_ix = torch.masked_select(gold[:,1:], mask[:,1:].byte())
                
                ix2label = {v:k for k, v in label2ix.items()}
                
                bpe_tok = [tokenizer.convert_ids_to_tokens([ix])[0] for ix in t_masked_ix.tolist()]
                
                flat_ori_tok = [item for sublist in ti for item in sublist]
                   
                deunk_bpe_tok = explore_unk(bpe_tok, flat_ori_tok)
                                
                for t, g, p in zip(deunk_bpe_tok, gold_masked_ix.tolist(), pred_masked_ix.tolist()):
                    fo.write('%s\t%s\t%s\n' % (t, ix2label[g], ix2label[p]))
                fo.write('EOR\tO\tO\n')


def eval_lab(model, test_dataloader):
    
    pred_np, lab_np = [], []
    model.eval()
    with torch.no_grad():
        for b_tok, b_mask, b_sour_mask, b_targ_mask, b_sent_mask, b_lab in test_dataloader:
            b_pred_prob = model(b_tok, b_sour_mask, b_targ_mask, token_type_ids=b_sent_mask, attention_mask=b_mask)
            b_pred = torch.argmax(b_pred_prob, dim=-1).squeeze(1)
            pred_np += b_pred.tolist()
            lab_np += b_lab.tolist()
    p, r, f1, _ = precision_recall_fscore_support(lab_np, pred_np, average='weighted')
    acc = accuracy_score(lab_np, pred_np)
    return p, r, f1, acc


def eval_seq(model, tokenizer, test_data, deunk_toks, label2ix, epoch, file_out):
    model.eval()
    with torch.no_grad():
        with open('outputs/%s_out.txt' % file_out, 'w') as fo, open('outputs/%s_eval.txt' % file_out, 'w') as fe:
            for deunk_tok, (token, mask, gold) in zip(deunk_toks, test_data):
                pred_prob = model(token, attention_mask=mask)
                pred = torch.argmax(pred_prob, dim=-1)

                t_masked_ix = torch.masked_select(token[:,1:], mask[:,1:].byte())
                pred_masked_ix = torch.masked_select(pred[:,1:], mask[:,1:].byte())
                gold_masked_ix = torch.masked_select(gold[:,1:], mask[:,1:].byte())
                
                ix2label = {v:k for k, v in label2ix.items()}
                
                bpe_tok = [tokenizer.convert_ids_to_tokens([ix])[0] for ix in t_masked_ix.tolist()]
                
                assert len(bpe_tok) == len(deunk_tok)
                
                for t, g, p in zip(deunk_tok, gold_masked_ix.tolist(), pred_masked_ix.tolist()):
                    fe.write('%s\t%s\t%s\n' % (t, ix2label[g], ix2label[p]))
                fe.write('\n')
                for dt, t, g, p in zip(deunk_tok, bpe_tok, gold_masked_ix.tolist(), pred_masked_ix.tolist()):
                    fo.write('%s\t%s\t%s\t%s\n' % (dt, t, ix2label[p], '_'))
                fo.write('\n')
