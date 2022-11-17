#!/usr/bin/env python3
# -*- coding: utf8 -*-

"""
Utility functions used for Logic & Language course at Utrecht University
contact: Lasha.Abzianidze@uu.nl
"""

from typing import List
import nltk

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
