import torch
from collections import defaultdict


def transformer_word2convec(model, tokenizer, word_list_batch, v=False, device=torch.device("cpu"), layer=-1, collate_tok_vec=torch.mean):
    """ Uses tokenizer and model to assign vectors to words in a batch of tokenized text.
        If a word is represented by several tokens, then the vectors of the tokens are collated with mean (by default).
        @layer indicates the layer from which vectors are extracted.
        return
            List[List[Dict{"word"->word_token, "tokens"->List[tokens], "pt"->pytorch tensor}]]
    """
    model.config.output_hidden_states = True
    if device: model.to(device)
    vectorized = []
    batch_size = len(word_list_batch)
    with torch.no_grad():
        tokenized = tokenizer(word_list_batch, padding=True, return_tensors='pt', is_split_into_words=True).to(device)
        if v: print(tokenized)
        if v: print(tokenized.input_ids)
        token_list_batch = [ tokenizer.convert_ids_to_tokens(tokenized.input_ids[i], skip_special_tokens=True) \
                        for i in range(batch_size) ]
        if v: print(token_list_batch)
        # return tokenized
        # print(type(tokenized))

        # mapping token positions to word positions
        # helped from https://discuss.huggingface.co/t/issue-with-extracting-word-ids-from-batch-encoding-object/20280
        word_ids_batch = [ tokenized[i].word_ids for i in range(batch_size) ]
        # print(word_ids_batch)
        tok_pos_to_word_pos = [ [ i for i in word_ids if i is not None ] for word_ids in word_ids_batch ]
        if v: print(tok_pos_to_word_pos)
        # mapping word positions to token positions
        word_pos_to_tok_pos = []
        for i in range(batch_size):
            mapping = defaultdict(list)
            for tok_pos, word_pos in enumerate(tok_pos_to_word_pos[i]):
                mapping[word_pos].append(tok_pos)
            word_pos_to_tok_pos.append(dict(mapping))
        if v: print(word_pos_to_tok_pos)

        out = model(**tokenized)
        # make one big tensor with dimensions: batch, position, layer, vector dim
        hidden_states = torch.stack(out.hidden_states, dim=0).permute(1,2,0,3)  

        for i in range(batch_size):
            seq_vectorized = []
            for j, word in enumerate(word_list_batch[i]):
                word_tokens = [ token_list_batch[i][k] for k in word_pos_to_tok_pos[i][j] ]
                word_vectors_stacked = torch.stack([ hidden_states[i][k][layer] for k in word_pos_to_tok_pos[i][j]])
                word_vector = collate_tok_vec(word_vectors_stacked, dim=0)
                d = { 'word': word, 'tokens': word_tokens, 'pt': word_vector.cpu() }
                seq_vectorized.append(d)
            vectorized.append(seq_vectorized)
    return vectorized
