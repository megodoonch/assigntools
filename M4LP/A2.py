import subprocess
import os
from os import path as op
import re
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

def taged2offsets(tag, s):
    """ Takes a sentence that includes words marked with <tag></tag>
        and returns a tag-free sentence along with character offsets
        of the marked words, so that it is still recoverable which words
        were tagged in the original input. 
    """
    offsets = []
    tags_length = 5 + 2*len(tag) # 5 because of length of "<></>""
    removed_tags_length = 0 # no tags are removed yet
    for r in re.finditer(f"<{tag}>(.+?)</{tag}>", s):
        # when tags are removed, the end of the target word additionally 
        # moves front with the number of positions equal to removed tag's length
        # but at the same time, all following offsets should move with positions
        # equal to scanned tags  
        start = r.start() - removed_tags_length
        end = r.end() - tags_length - removed_tags_length
        offsets.append((start, end))
        removed_tags_length += tags_length
    # now we are actually removing tags from teh original tagged sentence 
    cleaned_s = re.sub(f"</?{tag}>", "", s)
    return cleaned_s, offsets


def evaluate_contextual_lex_rel(pred, data, draw=False):
    '''Compare predictions to gold standard'''
    mapping = {'equivalence': 'equi', 'other-related': 'othr', 
               'reverse_entailment':'r-ent', 'alternation':'alter', 
               'independent':'ind', 'forward_entailment':'f-ent'}
    gold = [ mapping[d['r']] for d in data ] # gold relations
    pred = [ mapping[p] for p in pred ]  
    if len(pred) != len(gold):
        raise RuntimeError(f'Length of predictions({len(pred)}) and gold relations({len(gold)}) do not match')
    labels = sorted(set(gold))
    acc = accuracy_score(gold, pred)
    if draw == True:
        cm = confusion_matrix(gold, pred, labels=labels)
        m = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        m.plot()
    return acc


def run_cmd(cmd, v=False):
    """ run cmd and print stdout lines while the command is running if v is True.
        Return the stdout as a string
    """
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                            stderr=subprocess.STDOUT, text=True, shell=True)
    out_str = ''  
    while(True):
        out = p.poll() 
        line = p.stdout.readline()
        if line: 
            out_str += line
            if v: print(line, end="")
        if out is not None:
            break
    return out_str

def show_tableau(filename, tableau_css='/content/LangPro/css/tableau.css'):
    """ Intended for google colab use
    """
    import IPython
    import google.colab.output
    if not op.isfile(tableau_css):
        raise RuntimeError(f"File {tableau_css} cannot be found")
    google.colab.output._publish.css(open(tableau_css).read())
    return IPython.display.HTML(filename=filename)

###############################################################
# Auxiliary functions for sanity checking parameters
def check_align_param(align):
    # process alignmnet
    align_vals = "align no_align both".split()
    assert align in align_vals, f"align={align} should be of one {align_vals}"

def check_pids_param(pids):
    if pids is None or pids=='all': pids = '_'
    elif isinstance(pids, list) or isinstance(pids, str): pass
    else: raise RuntimeError(f"problem ids should be a string, list or None: {pids}\n"
        f"Examples of accepted format: None/'all' = all problems, '1-100' = problesm from 1 to 100, [1,20, 21] - only problems 1, 20 & 21"
    )
    return pids 

def check_annos_param(annos):
    annos['ner'] = "'cc2016.st'"
    return ', '.join([ f"{a}-{t}" for a, t in annos.items() ])


class LangPro:
    def __init__(self, prover_dir, data_dir):
        self.main_pl = op.abspath(op.join(prover_dir, "prolog/main.pl"))
        self.wn_pl = op.abspath(op.join(prover_dir, "WNProlog/wn.pl"))
        self.dir = op.abspath(prover_dir)
        # check if files exist
        assert op.isfile(self.main_pl), f"{self.main_pl} is not a file"
        assert op.isfile(self.wn_pl), f"{self.wn_pl} is not a file"
        self.parses = dict()
        possible_parsers = "cc2016 easyccg depccg".split()
        for f in os.listdir(data_dir):
            if f.endswith('anno_sen.pl'): 
                self.tok_anno_pl = op.abspath(op.join(data_dir, f))
            elif f.endswith('_sen.pl'): 
                self.sen_pl = op.abspath(op.join(data_dir, f)) 
            else:
                for p in possible_parsers:
                    if f"_{p}" in f: 
                        self.parses[p] = op.abspath(op.join(data_dir, f))
                        break
        for a in ["tok_anno_pl", "sen_pl"]:
            if a not in dir(self): 
                raise RuntimeError(f"{a} attribute was not initialized") 

    #################################################################

    # classifying a collection of NLI problems
    def nli_prove(self, pids, v=False, print_prob=False,
            parallel=False, parts=['train','trial'], ral=50, kb=None,
            labels=['yes', 'no', 'unknown'], align="align", 
            annos={'ccg':'depccg', 'l':'spacy', 'ppos':'spacy'}):
        """ self: langpro object
            v: print whatever is output to teh stdout
            print_prob: print sentecnes of a problem for each processed problem,
                bt default, only sentences of unsolved problems are printed
            pids: a list of problem ids or an interval 'X-Y', None=all
            parts: filter problems based on the partition names trial, train, test
            ral: rule application limit
            out: if output needs to be written in a file
            labels: filter problems based on the gold labels
            annos: a dictionary of an annotation layer and tool name used for processing
            kb: a string representing a sequence of relations
            align: 'align' and 'no_align' considers only aligned and non-aligned terms,
                respectively. Use 'both' to conside both modes (but slows things by ~40%).
            returns a dictionary of problem ids and predictions
        """
        # process parameters
        pids = check_pids_param(pids)
        check_align_param(align)
        anno_sys = check_annos_param(annos)
        # process other args
        conccurent = "parallel(_), " if parallel else ""
        prprb = 'prprb, ' if print_prob else ''
        kb = kb if kb else ''
        # put all in a single command line
        cmd = f"""swipl -g "parList([ {prprb}parts({parts}), aall, allInt, constchk, {conccurent}wn,\
            ral({ral}), llf_norm, ccg_norm,\
            anno_sys([{anno_sys}]) ]), prob_input_to_list({pids}, PIDs), \
            prove_nli(PIDs, {align}, [{kb}], {labels}), \
            halt" \
            -f {self.main_pl} -l {self.tok_anno_pl} {self.sen_pl} {self.parses[annos['ccg']]} {self.wn_pl}
            """
        out = run_cmd(cmd, v=v)
        return out

    # tableau prove a particular problem and display its tableau proof
    def tableau_prove(self, pid, v=False, ral=50, kb=None,
            align="align", annos={'ccg':'depccg', 'l':'spacy', 'ppos':'spacy'}):
        """ self: langpro object
            pid: a problem id
            ral: rule application limit
            annos: a dictionary of an annotation layer and tool name used for processing
            kb: a string representing a sequence of relations
            align: 'align' and 'no_align' considers only aligned and non-aligned terms,
                respectively. Use 'both' to conside both modes (but slows things by ~30%).
            returns inference label and draw a tableau proof
        """
        # process parameters
        check_align_param(align)
        anno_sys = check_annos_param(annos)     
        # process other args
        pid = int(pid)
        kb = kb if kb else ''
        # put all in a single command line
        cmd = f"""cd {self.dir} && swipl -g "parList([ parts([train,trial,test]), aall, allInt, constchk, wn,\
            ral({ral}), llf_norm, ccg_norm, html, proof_tree, anno_sys([{anno_sys}]) ]), \
            prove_nli({pid}, {align}, [{kb}]), \
            halt" \
            -f {self.main_pl} -l {self.tok_anno_pl} {self.sen_pl} {self.parses['depccg']} {self.wn_pl}
            """
        out = run_cmd(cmd, v=v)
        align_modes = ['align', 'no_align'] if align == 'both' else [align] 
        modes = [ (m, yn) for m in align_modes for yn in ['yes', 'no'] ]
        mode2html = { (m, yn):op.join(self.dir, f'xml/tableau-{pid}-{yn}-{m}.html') for m, yn in modes }
        return out, mode2html
