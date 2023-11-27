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
def prover9_prove(path: str, conclusion: str, premises: List[str] = []) -> bool:
    """
    Given a conclusion and a list of premises, 
    tries to prove whether the premises entail the conclusion.
    Returns a boolean value indicating whether the proof was found
    """
    str2exp = nltk.sem.Expression.fromstring
    c = str2exp(conclusion)
    ps = [ str2exp(p) for p in premises ]
    prover9 = nltk.Prover9()
    if path: prover9.config_prover9(path)
    return prover9.prove(c, ps)

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

#########################################################
# follows terminology at https://en.wikipedia.org/wiki/Syllogism
def gen_syllogism(M, S, P, neg="not", types="aeio", figures="1234"):
    """
    Returns a generator that produces syllogisms. Each syllogism is a pair of
    its id (consisting of the figure and sentence types) and a tuple of three
    sentences (2 premises and a conclusion).
    """
    all_figures = {'1':('MP','SM'), '2':('PM','SM'), '3':('MP','MS'), '4':('PM','MS')}
    all_types = {'a': "All {} are {}",
                 'e': "No {} are {}",
                 'i': "Some {} are {}",
                 'o': f"Some {{}} are {neg} {{}}"
    }
    # filter the figures and types based on the input
    sel_figures = tuple( (f, v) for (f, v) in sorted(all_figures.items()) if f in figures )
    sel_types = tuple( (t, v) for (t, v) in sorted(all_types.items()) if t in types ) 

    d = {'M':M, 'S':S, 'P':P}
    # generation by looping over figures
    for f, (p1, p2) in sel_figures:
        # make placeholders for f-string substitutions 
        p1 = [ f"{{{x}}}" for x in p1 ]
        p2 = [ f"{{{x}}}" for x in p2 ]
        p = [ '{S}', '{P}' ]
        # generate sentences by looping over sentence types
        for t, s in sel_types:
            for t1, s1 in sel_types:
                for t2, s2 in sel_types:
                    prem1 = s1.format(*p1)
                    prem2 = s2.format(*p2)
                    con = s.format(*p)
                    yield f"f{f}-{t1}{t2}{t}", (prem1.format(**d), prem2.format(**d), con.format(**d))

