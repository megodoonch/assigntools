#!/usr/bin/env python3
# -*- coding: utf8 -*-

import numpy as np
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import typing

def get_gpu_info():
    """
    Return device (GPU or CPU) depending on a GPU availability
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available')
        print(f"GPU's name is {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(device)})")
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device

def load_tok_model(hub_name):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(hub_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(hub_name)
    return tokenizer, model

def probs2prediction(probs, id2label):
    """ 
    Gets prob distribution and selects the max with its corresponding label.
    Returns dict of prediction details
    """
    lab_index = np.argmax(probs)
    return {"label_index": lab_index, "label": id2label[lab_index],
            "prob": probs[lab_index],
            "probs": {l:probs[i] for i, l in id2label.items()}}

def predict_nli(tokenizer, model, nli_prob, device=None):
    """ 
    nli_prob - list with two elements
    """
    encoded_prob = tokenizer(*nli_prob, truncation=True, padding=True, return_tensors="pt")
    encoded_prob = encoded_prob.to(device) if device else encoded_prob.to(model.device)
    output = model(**encoded_prob) #transformers.modeling_outputs.SequenceClassifierOutput
    probs = torch.softmax(output.logits, dim=1).tolist()[0]
    return probs2prediction(probs, model.config.id2label)

def batch_predict_nli(tokenizer, model, nli_list, batch_size=32, device=None):
    start, predictions = 1, []
    batch_probs = nli_list[(start-1)*batch_size : start*batch_size]
    while batch_probs:
        print('.', end='')
        batch_probs_enc = tokenizer(batch_probs, padding=True, truncation=True, return_tensors="pt")
        batch_probs_enc = batch_probs_enc.to(device) if device else batch_probs_enc.to(model.device)
        out = model(**batch_probs_enc)
        pred_prob_list = torch.softmax(out.logits, dim=1).tolist()
        predictions += [ probs2prediction(probs, model.config.id2label) for probs in pred_prob_list ]
        start += 1
        batch_probs = nli_list[(start-1)*batch_size : start*batch_size]
    return predictions
