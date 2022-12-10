#!/usr/bin/env python3
# -*- coding: utf8 -*-

import spacy

tokenized2Doc(raw, tokens, spacy_pipeline):
    """
    Takes raw text and its tokenized version and returns spaCy's Doc object 
    """
    # TODO: initialize spaces arg too, now it defults to the list of True
    return Doc(spacy_pipeline.vocab, words=tokens)
