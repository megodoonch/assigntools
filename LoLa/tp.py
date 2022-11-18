#!/usr/bin/env python3
# -*- coding: utf8 -*-

"""
Utility functions used for Logic & Language course at Utrecht University
contact: Lasha.Abzianidze@uu.nl
"""

from typing import List, Tuple, Dict
import nltk
import re

#########################################################
def tableau_prove(conclusion: str, premises: List[str] = [], verbose: bool = False) -> bool:
    """ 
    Given a conclusion and a list of premises, builds a tableau and
    detects whether the premises entail the conclusion.
    Returns a boolean value and optionally prints the tableau structure
    """
    str2exp = nltk.sem.Expression.fromstring
    c = str2exp(conclusion)
    ps = [ str2exp(p) for p in premises ] 
    return nltk.TableauProver().prove(c, ps, verbose=verbose)

#########################################################
def make_vars_consistent(prop_maps: List[Tuple[str, Dict[str,str]]], 
                         prop_letter: str = 'Q') \
    -> Tuple[List[str], Dict[str,str]]:
    """
    Takes a list of pairs of a propositional formuals and the mapping from
    propositional letters to the natural language sentences.
    It renames all propositional letters in the formulas and makes sure that 
    the mapping from letter to sentences is one to one.
    """
    all_sents = set([ v for (_, m) in prop_maps for k, v in m.items() ])
    sent2index = { s: i for i, s in enumerate(sorted(all_sents), start=1) }
    mapping, props = dict(), []
    for prop, m in prop_maps:
        for pi, sent in m.items():
            qi = f"{prop_letter}{sent2index[sent]}"
            prop = re.sub(fr'\b{pi}\b', qi, prop)
        props.append(prop)
        mapping[qi] = sent
    return props, mapping    



#########################################################
def tableau_equiv(p: str, q: str) -> bool:
    """
    Based on an NLTK tableau, check whether two formulas are equivalent
    """
    return tableau_prove(f"({p}) <-> ({q})", [])



#########################################################
def prop_entail(sent2prop, premises: List[str], conclusion: str, verbose: bool = False) -> bool:
    """
    Given a conclusion sentence and a (possibly empty) set of premises (in natural language),
    Detect whether the premises entail the conclusion, i.e., the conclusion 
    is uninformative wrt the premises. This is done via translation into PL.
    """
    sents = premises + [conclusion]
    prop_maps = [ sent2prop(s) for s in sents ]
    props, mapping = make_vars_consistent(prop_maps)
    if verbose:
        print(f"{props[:-1]} =?=> {props[-1]}")
        print(mapping)
    return tableau_prove(props[-1], premises=props[:-1], verbose=verbose)

