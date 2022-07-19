def truncate_encoded_text(encoded_text, mode='first', is_bart_encoding=False):
    for k in encoded_text.keys():
        if mode == 'first':
            encoded_text[k] = encoded_text[k][:512]
        elif mode == 'last':
            if is_bart_encoding:
                encoded_text[k] = encoded_text[k][-512:]
            else:
                # The encoding was done by BERT tokenizer and the first token is always the cls one
                encoded_text[k] = [encoded_text[k][0]] + encoded_text[k][-511:]
        else:
            encoded_text[k] = encoded_text[k][:256] + encoded_text[k][-256:]
    return encoded_text