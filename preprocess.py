import os
from collections import defaultdict
import xml.etree.ElementTree as ET

# third-part libs
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_pretrained_bert import BertTokenizer
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit, KFold
from pyknp import Juman
import mojimoji
import utils
# from utils import *

juman = Juman()

import logging
logger = utils.init_logger('Data_Preprocess', logging.INFO, logging.INFO)

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
# BERT_URL='/Users/fei-c/Resources/embed/L-12_H-768_A-12_E-30_BPE'
# if str(device) == 'cuda':
#     BERT_URL='/larch/share/bert/Japanese_models/Wikipedia/L-12_H-768_A-12_E-30_BPE'
# tokenizer = BertTokenizer.from_pretrained(BERT_URL, do_lower_case=False, do_basic_tokenize=False)

merge_map_6c = {
    'after': 'AFTER',
    'met-by': 'AFTER',
    'overlapped-by': 'OVERLAP-OR-AFTER',
    'finishes': 'OVERLAP-OR-AFTER',
    'during': 'OVERLAP',
    'started-by': 'OVERLAP',
    'equal' : 'OVERLAP',
    'starts': 'BEFORE-OR-OVERLAP',
    'contains': 'OVERLAP',
    'finished-by' : 'OVERLAP',
    'overlaps' : 'BEFORE-OR-OVERLAP',
    'meets' : 'BEFORE',
    'before': 'BEFORE',
    'is_included' : 'OVERLAP',
    'identity' : 'OVERLAP',
    'includes' : 'OVERLAP',
    'vague' : 'VAGUE',
}

merge_map_4c = {
    'after': 'AFTER',
    'met-by': 'AFTER',
    'overlapped-by': 'OVERLAP',
    'finishes': 'OVERLAP',
    'during': 'OVERLAP',
    'started-by': 'OVERLAP',
    'equal' : 'OVERLAP',
    'starts': 'OVERLAP',
    'contains': 'OVERLAP',
    'finished-by' : 'OVERLAP',
    'overlaps' : 'OVERLAP',
    'meets' : 'BEFORE',
    'before': 'BEFORE',
    'is_included' : 'OVERLAP',
    'identity' : 'OVERLAP',
    'includes' : 'OVERLAP',
    'vague' : 'VAGUE',
}


def get_lab2ix_from_type(lab_type):
    lab2ix = {}
    if lab_type == '6c':
        for key in merge_map_6c.values():
            if key not in lab2ix:
                lab2ix[key] = len(lab2ix)
        return lab2ix
    elif lab_type == '4c':
        for key in merge_map_4c.values():
            if key not in lab2ix:
                lab2ix[key] = len(lab2ix)
        return lab2ix
    else:
        for key in merge_map_6c.keys():
            if key not in lab2ix:
                lab2ix[key] = len(lab2ix)
        return lab2ix


def count_major(labs):
    lab2count = {}
    for l in labs:
        if l not in lab2count:
            lab2count[l] = 1
        else:
            lab2count[l] += 1
    return lab2count, max(lab2count, key=lab2count.get), 100 * max(lab2count.values()) / sum(lab2count.values())


def merge_lab(lab, lab_type='6c'):
    # Goal: merge 13+1 labels into 5+1 or 3+1 labels.
    if lab_type == '6c':
        return merge_map_6c[lab] if lab in merge_map_6c else 'VAGUE'
    elif lab_type == '4c':
        return merge_map_4c[lab] if lab in merge_map_4c else 'VAGUE'


def vote_labels(lab_a, lab_b, lab_c, lab_type=None, comp=None):
    if lab_type in ['4c', '6c']:
        lab_a = merge_lab(lab_a, lab_type=lab_type)
        lab_b = merge_lab(lab_b, lab_type=lab_type)
        lab_c = merge_lab(lab_c, lab_type=lab_type)

    if comp:
        if lab_a == lab_b == lab_c:
            return lab_a
        else:
            return None
    else:
        if lab_a == lab_b or lab_a == lab_c:
            return lab_a
        elif lab_b == lab_c:
            return lab_b
        else:
            return 'VAGUE'


def mask_mention(sent_mask, mention_id):
    return [1 if mask == mention_id else 0 for mask in sent_mask]


def flatten_tlink_instance(deunk_toks, toks, sour_masks, targ_masks, labs):
    f_deunk_toks, f_toks = [], []
    for det, t, ls in zip(deunk_toks, toks, labs):
        f_deunk_toks += [det] * len(ls)
        f_toks += [t] * len(ls)
    f_sour_masks = [mask for sent in sour_masks for mask in sent]
    f_targ_masks = [mask for sent in targ_masks for mask in sent]
    f_labs = [mask for sent in labs for mask in sent]
    assert len(f_deunk_toks) == len(f_toks) == len(f_sour_masks) == len(f_targ_masks) == len(f_labs)
    return f_deunk_toks, f_toks, f_sour_masks, f_targ_masks, f_labs


def merge_word_mention_boundaries(flat_word_ids, flat_doc_toks, mention_offsets):
    merge_num = 0

    flat_new_word_ids = flat_word_ids.copy()

    tmp_offs = 0

    for mid, mtype, offs_b, offs_e, m in mention_offsets:
        print(merge_num, mid, mtype, offs_b, offs_e, m)
        if offs_b == 0:
            continue

        if flat_word_ids[offs_b - 1] == flat_word_ids[offs_b]:
            print('b', m, flat_doc_toks[offs_b], flat_doc_toks[offs_b - 1],
                  offs_b, offs_b - 1,
                  flat_word_ids[offs_b - 1], flat_word_ids[offs_b])
            while tmp_offs < offs_b:
                #                 print(tmp_offs, flat_doc_toks[tmp_offs], flat_word_ids[tmp_offs])
                flat_new_word_ids[tmp_offs] = flat_word_ids[tmp_offs] + merge_num
                print(flat_word_ids[tmp_offs] + merge_num)
                tmp_offs += 1
            merge_num += 1

        if flat_word_ids[offs_e - 1] == flat_word_ids[offs_e]:
            print('e', m, flat_doc_toks[offs_e], flat_doc_toks[offs_e - 1],
                  offs_e, offs_e - 1,
                  flat_word_ids[offs_e - 1], flat_word_ids[offs_e])
            while tmp_offs < offs_e:
                #                 print(tmp_offs, flat_doc_toks[tmp_offs], flat_word_ids[tmp_offs])
                flat_new_word_ids[tmp_offs] = flat_word_ids[tmp_offs] + merge_num
                tmp_offs += 1
            merge_num += 1
        print(flat_word_ids[tmp_offs - 1 if tmp_offs > 0 else 0] + merge_num)
        print()
    while tmp_offs < len(flat_word_ids):
        #         print(tmp_offs)
        flat_new_word_ids[tmp_offs] = flat_word_ids[tmp_offs] + merge_num
        tmp_offs += 1

    assert len(flat_word_ids) == len(flat_new_word_ids)

    return flat_new_word_ids


def attach_word_ids(doc_words):
    word_ids = []
    tmp_begin_id = 0
    for sent_word in doc_words:
        sent_ids = []
        for word in sent_word:
            sent_ids += [tmp_begin_id] * len(list(word))
            tmp_begin_id += 1
        word_ids.append(sent_ids)
    return word_ids


def retrieve_mention(sent_toks, ment_mask):
    assert len(sent_toks) == len(ment_mask)
    return [[t] for t, m in zip(sent_toks, ment_mask) if m == 1]


def extract_sents_from_xml_v2(xml_file, tokenizer, lab_type=None, comp=None):
    doc_deunk_toks, doc_toks = [], []
    eiid2eid = {}
    doc_mid2smask = {}
    doc_tlinks = defaultdict(lambda: list())
    root = ET.parse(xml_file).getroot()
    for text_node in root.findall('TEXT'):

        for dct_node in text_node.find('TIMEX3'):
            pass

        for event_node in text_node.iter('MAKEINSTANCE'):
            eiid2eid[event_node.attrib['eiid']] = event_node.attrib['eventID']
        logger.debug('event num: %i' % len(eiid2eid))

        for tlink_node in text_node.iter('TLINK'):
            if tlink_node.attrib['task'] in ['DCT', 'T2E']:
                sour_mid = tlink_node.attrib['timeID']
                targ_mid = eiid2eid[tlink_node.attrib['relatedToEventInstance']]
            elif tlink_node.attrib['task'] in ['E2E', 'MAT']:
                sour_mid = eiid2eid[tlink_node.attrib['eventInstanceID']]
                targ_mid = eiid2eid[tlink_node.attrib['relatedToEventInstance']]
            else:
                raise Exception("[ERROR] Unknown 'task' setting...")
            voted_label = vote_labels(
                tlink_node.attrib['relTypeA'],
                tlink_node.attrib['relTypeB'],
                tlink_node.attrib['relTypeC'],
                lab_type=lab_type,
                comp=comp
            )
            if sour_mid and targ_mid and voted_label:
                doc_tlinks[tlink_node.attrib['task']].append((sour_mid, targ_mid, voted_label))

        s_id = 0
        for sent_node in text_node.iter('sentence'):
            logger.debug('sentence %i' % s_id)
            sent_toks, tmp_mids = [], []
            for tag in sent_node.iter():
                try:
                    if tag.text and tag.text.strip():
                        text_seg = [w.midasi for w in juman.analysis(
                            mojimoji.han_to_zen(tag.text.strip()).replace('\u3000', '[JSP]')).mrph_list()]
                        if tag.tag in ['EVENT', 'event'] and 'eid' in tag.attrib:
                            tmp_mids.append(tag.attrib['eid'])
                            doc_mid2smask[tag.attrib['eid']] = [
                                s_id,
                                [0] * len(sent_toks) + [1] * len(text_seg)
                            ]
                        elif tag.tag in ['TIMEX3'] and 'tid' in tag.attrib:
                            tmp_mids.append(tag.attrib['tid'])
                            doc_mid2smask[tag.attrib['tid']] = [
                                s_id,
                                [0] * len(sent_toks) + [1] * len(text_seg)
                            ]
                        sent_toks += text_seg
                    if tag.tag != 'sentence' and tag.tail and tag.tail.strip():
                        tail_seg = [w.midasi for w in juman.analysis(
                            mojimoji.han_to_zen(tag.tail.strip()).replace('\u3000', '[JSP]')).mrph_list()]
                        sent_toks += tail_seg
                except Exception as ex:
                    logger.error(xml_file, tag.tag, tag.text, tag.attrib)

            """ subword tokenizer for word tokens """
            sbp_toks = tokenizer.tokenize(' '.join(sent_toks))
            deunk_toks = utils.explore_unk(sbp_toks, sent_toks)
            logger.debug(str(len(sent_toks)) + ' ' + '/'.join(sent_toks))
            logger.debug(str(len(deunk_toks)) + ' ' + '/'.join(deunk_toks))

            """ padding sentence mention masks with matching sbp tokens """
            for mid in tmp_mids:
                sent_mask = utils.padding_1d(doc_mid2smask[mid][1], len(sent_toks))
                logger.debug('%s, sent_id: %i' % (mid, doc_mid2smask[mid][0]))
                logger.debug(str(len(sent_mask)) + ' ' + ' '.join([str(i) for i in sent_mask]))
                sbp_mask = utils.match_sbp_mask(sbp_toks, sent_mask)
                doc_mid2smask[mid][1] = sbp_mask
                logger.debug(str(len(sbp_mask)) + ' ' + ' '.join([str(i) for i in sbp_mask]))
                logger.debug(retrieve_mention(deunk_toks, sbp_mask))

            logger.debug('[EOS]')

            if deunk_toks:
                doc_deunk_toks.append(deunk_toks)
                doc_toks.append(sbp_toks)
                s_id += 1

    return doc_deunk_toks, doc_toks, doc_mid2smask, doc_tlinks


def make_tlink_instances_v2(doc_deunk_toks, doc_toks, doc_mid2smask, doc_tlinks, task=None):
    deunk_toks, toks, sour_masks, targ_masks, sent_masks, mids, rels = [], [], [], [], [], [], []
    for sour_mid, targ_mid, rel in doc_tlinks[task]:
        logger.debug('%s\t%s\t%s' % (sour_mid, targ_mid, rel))
        targ_sid = doc_mid2smask[targ_mid][0]
        if task in ['DCT']:
            deunk_tok = doc_deunk_toks[targ_sid]
            tok = doc_toks[targ_sid]
            sour_mask = [0] * len(doc_mid2smask[targ_mid][1])
            targ_mask = doc_mid2smask[targ_mid][1]
            sent_mask = [0] * len(doc_mid2smask[targ_mid][1])
        elif task in ['T2E', 'E2E', 'MAT']:
            if sour_mid not in doc_mid2smask:
                continue
            sour_sid = doc_mid2smask[sour_mid][0]
            if targ_sid - sour_sid == 0:
                deunk_tok = doc_deunk_toks[targ_sid]
                tok = doc_toks[targ_sid]
                sour_mask = doc_mid2smask[sour_mid][1]
                targ_mask = doc_mid2smask[targ_mid][1]
                sent_mask = [0] * len(doc_mid2smask[targ_mid][1])
            else:
                deunk_tok = doc_deunk_toks[sour_sid] + doc_deunk_toks[targ_sid]
                tok = doc_toks[sour_sid] + doc_toks[targ_sid]
                sour_mask = doc_mid2smask[sour_mid][1] + [0] * len(doc_mid2smask[targ_mid][1])
                targ_mask = [0] * len(doc_mid2smask[sour_mid][1]) + doc_mid2smask[targ_mid][1]
                sent_mask = [0] * len(doc_mid2smask[sour_mid][1]) + [1] * len(doc_mid2smask[targ_mid][1])


        logger.debug(' '.join(deunk_tok))
        logger.debug(' '.join(tok))
        logger.debug(' '.join([str(i) for i in sour_mask]))
        logger.debug(' '.join([str(i) for i in targ_mask]))
        logger.debug(' '.join([str(i) for i in sent_mask]))
        deunk_toks.append(deunk_tok)
        toks.append(tok)
        sour_masks.append(sour_mask)
        targ_masks.append(targ_mask)
        sent_masks.append(sent_mask)
        rels.append(rel)
        assert len(deunk_tok) == len(tok) == len(sour_mask) == len(targ_mask) == len(sent_mask)
    return deunk_toks, toks, sour_masks, targ_masks, sent_masks, rels


def batch_make_tlink_instances_v2(file_list, tokenizer, task=None, lab_type=None, comp=None):
    deunk_toks, toks, sour_masks, targ_masks, sent_masks, rels = [], [], [], [], [], []
    for dir_file in file_list:
        logger.debug('[Done] processing %s' % dir_file)
        doc_deunk_toks, doc_toks, doc_mid2smask, doc_tlinks = extract_sents_from_xml_v2(
            dir_file,
            tokenizer,
            lab_type=lab_type,
            comp=comp
        )
        inst_deunk_toks, inst_toks, inst_sour_masks, inst_targ_masks, inst_sent_masks, inst_rels = make_tlink_instances_v2(
            doc_deunk_toks,
            doc_toks,
            doc_mid2smask,
            doc_tlinks,
            task=task
        )
        deunk_toks += inst_deunk_toks
        toks += inst_toks
        sour_masks += inst_sour_masks
        targ_masks += inst_targ_masks
        sent_masks += inst_sent_masks
        rels += inst_rels
    return (deunk_toks, toks, sour_masks, targ_masks, sent_masks), rels


def convert_to_np_v2(deunk_toks, toks, sour_masks, targ_masks, sent_masks, labs, tokenizer, lab2ix):
    max_len = max([len(t) for t in toks])
    logger.info('max seq length %i' % (max_len))
    pad_tok_ids, pad_masks, pad_sm, pad_tm, pad_sent_m = [], [], [], [], []
    for inst_tok, inst_sm, inst_tm, inst_sent_m, inst_lab in zip(toks, sour_masks, targ_masks, sent_masks, labs):
        pad_inst_tok = utils.padding_1d(['[CLS]'] + inst_tok, max_len + 1, pad_tok='[PAD]')
        pad_inst_tok_ids = tokenizer.convert_tokens_to_ids(pad_inst_tok)
        pad_inst_masks = utils.padding_1d([1] * (len(inst_tok) + 1), max_len + 1, pad_tok=0)
        pad_inst_sm = utils.padding_1d([0] + inst_sm, max_len + 1, pad_tok=0)
        pad_inst_tm = utils.padding_1d([0] + inst_tm, max_len + 1, pad_tok=0)
        pad_inst_sent_m = utils.padding_1d([0] + inst_sent_m, max_len + 1, pad_tok=0)
        pad_tok_ids.append(pad_inst_tok_ids)
        pad_masks.append(pad_inst_masks)
        pad_sm.append(pad_inst_sm)
        pad_tm.append(pad_inst_tm)
        pad_sent_m.append(pad_inst_sent_m)
    lab_ids = [lab2ix[l] for l in labs]
    assert len(pad_tok_ids) == len(pad_masks) == len(pad_sm) == len(pad_tm) == len(pad_sent_m) == len(lab_ids)
    return np.array(pad_tok_ids), np.array(pad_masks), np.array(pad_sm), np.array(pad_tm), np.array(
        pad_sent_m), np.array(lab_ids)


def instance_to_tensors(deunk_toks, toks, sour_masks, targ_masks, sent_masks, labs, tokenizer, lab2ix, device):

    toks_ids_np, tok_masks_np, sour_masks_np, targ_masks_np, sent_masks_np, lab_ids_np = convert_to_np_v2(
        deunk_toks,
        toks,
        sour_masks,
        targ_masks,
        sent_masks,
        labs,
        tokenizer,
        lab2ix
    )

    tensors = TensorDataset(
        torch.from_numpy(toks_ids_np).to(device),
        torch.from_numpy(tok_masks_np).to(device),
        torch.from_numpy(sour_masks_np).to(device),
        torch.from_numpy(targ_masks_np).to(device),
        torch.from_numpy(sent_masks_np).to(device),
        torch.from_numpy(lab_ids_np).to(device),
    )

    return tensors


def doc_kfold(data_dir, cv=5):
    file_list, file_splits = [], []
    for file in sorted(os.listdir(data_dir)):
        if file.endswith(".xml"):
            dir_file = os.path.join(data_dir, file)
            file_list.append(dir_file)
    logger.info("[Number] %i files in '%s'" % (len(file_list), data_dir))
    gss = KFold(n_splits=cv, shuffle=True, random_state=1029)
    for train_split, test_split in gss.split(file_list):
        file_splits.append((
            [file_list[fid] for fid in train_split],
            [file_list[fid] for fid in test_split]
        ))
    return file_splits