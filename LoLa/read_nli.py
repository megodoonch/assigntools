import re, json, os
from os import path as op
from collections import Counter, defaultdict
from tqdm import tqdm

########################################################################
def snli_jsonl2dict(snli_dir, clean_labels=True):
    """"
    Reads jsonl files of snli parts and returns
    snli dict that contains problem level info: {part: {prob_id: Problem info}}
    sen2anno dict that contains sentence annotations: {sen:annotation dict}
    It is efficient to separate problem- and sentence-level info as
    many sentences reoccur in several problems.
    """"
    # Find .jsonl files corresponding to data parts
    PARTS = [ f[9:-6] for f in os.listdir(snli_dir) if re.match('.+_(train|dev|test).jsonl', f) ]
    if not PARTS:
        raise RuntimeError(f"No .jsonl files were found in {snli_dir}")
    else:
        print(f"Found .json files for {PARTS} parts")
    # Initializing values
    LABELS = 'entailment neutral contradiction'.split()
    snli = defaultdict(dict)
    sen2anno = defaultdict(dict) # maps sentence strings to its annotations 
    cleaned = 0
    label_set = set(LABELS)
    # Reading part files one by one
    for s in PARTS:
        weird_lab= {'cnt': Counter(), 'pids': []} # record problems with weird labels if any
        print("processing " + s.upper(), end=':\t')
        with open(op.join(snli_dir, f'snli_1.0_{s}.jsonl')) as F:
            for line in tqdm(F):
                prob = json.loads(line)
                # test if the problem has weird labels
                weird_labs = set(prob['annotator_labels']) - label_set
                if weird_labs:
                    weird_lab['cnt'].update(weird_labs)
                    weird_lab['pids'].append(prob['pairID'])
                if clean_labels and weird_labs: 
                    continue # ignore problems with weird labels
                # read a problem in a dict
                prob, p_anno, h_anno = json_prob2dict(prob, labels=LABELS)
                snli[s][prob['pid']] = prob
                # update sentences annotations
                sen2anno = update_sen2anno(sen2anno, prob['p'], p_anno, (s, prob['pid'], 'p'))
                sen2anno = update_sen2anno(sen2anno, prob['h'], p_anno, (s, prob['pid'], 'h'))
        print(f"{len(snli[s])} problems read")
        print(f"{len(weird_lab['pids'])} problems have a wrong annotator label")
    if weird_lab['cnt']:  
        most_common_weird = ','.join([ f"/{k}/({c})" for k, c in weird_lab['cnt'].most_common() ])
        print(f"Most common weird labels: {most_common_weird}")
    return snli, sen2anno


########################################################################
def update_sen2anno(sen2anno, sen, sen_anno, part_id_ph):
    """
    Updates sen2anno dictionary in place with sentence annotation and
    with the reference to the problem where it occurs in
    """
    if sen in sen2anno:
        sen2anno[sen]['pids'].add(part_id_ph)
    else:
        sen2anno[sen] = sen_anno
        sen2anno[sen]['pids'] = set([part_id_ph])
    return sen2anno


########################################################################
def json_prob2dict(prob, labels='entailment neutral contradiction'.split()):
    """
    Reprocess a prob dict and create a more informative dictionary 
    that records easily accessible info for an NLI problems.
    Problem info is seperated from premise/hypothesis annotations.
    The problem info contains the following info:
    a: authorlabel, g: gold label, pid: pair id, cid: caption ID, 
    lnum: label number, lcnt: Counter of annotation labels, 
    ltype: a string of three numbers corresponding to count of entailmen, neutral, contradiction annotations  
    h: hypothesis, p: premise.
    Premise and hypothesis annotation dict contain the info:
    tok: list of tokens in a sentence, pos: a list of pos tags of tokens,
    tree: a phrase structure tree of a sentence, btree: a binary phrase structure tree of a sentence. 
    """
    label_set = set(labels)
    p = { 'g':prob['gold_label'], 'pid':prob['pairID'], 'cid':prob['captionID'] }
    p['lnum'] = len([ l for l in prob['annotator_labels'] if l in label_set ])
    p['lcnt'] = Counter([ l for l in prob['annotator_labels'] if l in label_set ])
    p['ltype'] = ''.join([ str(p['lcnt'][l]) for l in labels ])
    p['p'], p['h'] = prob['sentence1'], prob['sentence2']
    # Premise sentence annotations
    p_anno = read_sentence_anno(prob['sentence1_parse'], prob['sentence1_binary_parse'])
    # Hypothesis sentence annotations
    h_anno = read_sentence_anno(prob['sentence2_parse'], prob['sentence2_binary_parse'])
    return p, p_anno, h_anno

########################################################################
def read_sentence_anno(tree, btree):
    """ 
    Read tokens and postags from trees.
    Also convert trees into NLTK tree objects.
    Return a dict {
        tok: a list of tokens, pos: a list of token postags 
        tree: tree as a string, btree: binary tree as a string }
    """
    anno = { 'tree': tree, 'btree': btree }
    anno['tok'] = [ t for t in re.split('[)( ]+', anno['btree']) if t ]
    anno['pos'] = re.findall('\(([^()]+)\s+[^()]+\)', anno['tree'])
    # check that the number of tokens coincides with the number of pos tags
    assert len(anno['pos']) == len(anno['tok']), f"{i}: len({anno['pos']}) =/= len({anno['tok']})"
    return anno
