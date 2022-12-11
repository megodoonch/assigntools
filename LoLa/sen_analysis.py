#!/usr/bin/env python3
# -*- coding: utf8 -*-

import spacy
import itertools
from tqdm import tqdm

def tokenized2Doc(spacy_pipeline, raw, tokens):
    """
    Takes raw text and its tokenized version and returns spaCy's Doc object 
    """
    # TODO: initialize spaces arg too, now it defults to the list of True
    return spacy.tokens.Doc(spacy_pipeline.vocab, words=tokens)


def spacy_sen_context(spacy_nlp, sen_context_dict, disable_components=[], n=0):
    """
    Takes spacy_nlp pipeline and processes sentences in sen_context_dict
    where the latter is {sen->context}, context carrying additional info about sentences.
    n - does cut off
    disable_components - a list of spacy pipeline components that will be disabled during processing.
    Returns sen->anno dict where anno has an additional key 'spacy' with value of spacy Doc
    Note that the function modifies sen_context_dict
    """
    if not n: n = len(sen_context_dict)
    doc_sen_anno = [ (tokenized2Doc(s, a['tok'], spacy_nlp), (s, a)) for s, a in itertools.islice(sen_context_dict.items(), n) ]
    sen_anno = dict() # a dictionary with sentence keys and anno gets additional key for 'spacy' that has spacy Doc as value
    with spacy_nlp.select_pipes(disable=disable_components):
        for doc, (sen, anno) in tqdm(spacy_nlp.pipe(doc_sen_anno, as_tuples=True)):
            sen_anno[sen] = anno
            sen_anno[sen]['spacy'] = doc
    return sen_anno


def spacy_process_sen2tok(spacy_nlp, sen2tok, disable_components=[]):
    """
    Takes spacy_nlp pipeline and processes sentences in sen2tok {sen->tok list} while adopting provided tokenization.
    disable_components - a list of spacy pipeline components that will be disabled during processing.
    Returns sen->Doc dict where Doc contains spacy analysis of sen
    """
    doc_sen = [ (tokenized2Doc(spacy_nlp, sen, tok), sen) for sen, tok in sen2tok.items() ]
    with spacy_nlp.select_pipes(disable=disable_components):
        return { sen: doc for doc, sen in tqdm(spacy_nlp.pipe(doc_sen, as_tuples=True)) }


def display_doc_dep(doc, d=150, compact=True, jupyter=True):
    """ A shortcut function for displaying spaCy dependencies.
        It uses compact representation by default.
    """
    spacy.displacy.render(doc, style='dep', jupyter=jupyter, \
                          options={'distance':d, 'fine_grained':True, 'compact':compact})
    
    
