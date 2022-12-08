import os
import sys
from os import path as op
from collections import Counter, defaultdict
import re
import nltk
import json
from nltk.metrics.agreement import AnnotationTask as AnnoT
from tabulate import tabulate
import pickle
import random

# read json file in dictionary using json.loads
# d[split][pairID] = {  a:author_label,  g:gold_label, pid: pair_id, cid: caption_ID 
#                       h:hypothesis, p:premise, 
#                       hl:hypothesis length, pl:premise length,
#                       ht:hypothesis tokenized, pt:premise tokenized
#                       n:label_num, lc:annotator_label_counter,
#                       lt:labeling_type, e.g., ECN-numbers}
#
def snli2dict(snli_dir, clean_labels=True):
    SPLIT = 'train dev test'.split()
    LABELS = 'entailment contradiction neutral'.split()
    snli = defaultdict(list)
    sen2pid = defaultdict(set)
    cleaned = 0
    label_set = set(LABELS)
    for s in SPLIT:
        weird_lab= {'cnt': Counter(), 'p': []}
        print("processing " + s.upper(), end=':\t')
        with open(op.join(snli_dir, f'snli_1.0_{s}.jsonl')) as F:
            for i, line in enumerate(F):
                prob = json.loads(line)
                # test if the problem has weird labels
                weird_labs = set(prob['annotator_labels']) - label_set
                if weird_labs:
                    weird_lab['cnt'].update(weird_labs)
                    weird_lab['p'].append(prob['pairID'])
                if clean_labels and weird_labs: 
                    continue # ignore problems with weird labels
                # define new problem item
                p = { 'g':prob['gold_label'], 'pid':prob['pairID'], 'cid':prob['captionID'] }
                p['p'], p['ptree'], p['pbtree'] = prob['sentence1'], prob['sentence1_parse'], \
                                            prob['sentence1_binary_parse']
                p['h'], p['htree'], p['hbtree'] = prob['sentence2'], prob['sentence2_parse'], \
                                            prob['sentence2_binary_parse']
                p['lnum'] = len([ l for l in prob['annotator_labels'] if l in LABELS ])
                p['lcnt'] = Counter([ l for l in prob['annotator_labels'] if l in LABELS ])
                p['ltype'] = ''.join([ str(p['lcnt'][l]) for l in LABELS ])
                p['ptok'] = [ t for t in re.split('[)( ]+', p['pbtree']) if t ]
                p['htok'] = [ t for t in re.split('[)( ]+', p['hbtree']) if t ]
                p['plen'], p['hlen'] = len(p['ptok']), len(p['htok'])
                p['ppos'] = re.findall('\(([^()]+)\s+[^()]+\)', p['ptree'])
                p['hpos'] = re.findall('\(([^()]+)\s+[^()]+\)', p['htree'])
                assert len(p['ppos']) == p['plen'], f"{i}: len({p['ppos']}) =/= {p['plen']}"
                assert len(p['hpos']) == p['hlen'], f"{i}: len({p['hpos']}) =/= {p['hlen']}"
                snli[s].append(p)
                # add sentences to sentence dict
                sen2pid[p['p']].add((s, p['pid']))
                sen2pid[p['h']].add((s, p['pid']))
        print(f"{len(snli[s])} problems read")
        print(f"{len(weird_lab['p'])} problems have a wrong annotator label")
    most_common_weird = ','.join([ f"/{k}/({c})" for k, c in weird_lab['cnt'].most_common() ])
    print(f"Most common weird labels /{most_common_weird}/")
    return snli, sen2pid
