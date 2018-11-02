import re
from pathlib import Path
import numpy as np
import torch

def preprocess(txt):
    
    '''Generic preprocessing: remove uninformative text, replace numbers,
    create end of sentence tokan, replace punctuation, collapse white space.'''
    
    punct = '!"#$%&\'()*+,-./:;=?@[\\]^_`{|}~'

    # Remove stuff inside brackets: <>
    # Brackets contain uninformative items for our classifier. 
    # Example: '<CHAPTER ID="008">\nCorrections to ...' 
    txt = re.sub('\<.*?\>', '', txt, count = 0)

    # Replace numbers with special character
    txt = re.sub('[0-9]+', ' <num> ', txt, count = 0)
    txt = txt.replace(' <num> .', ' <num> ')

    # Create special character for sentence end
    txt = txt.replace('.', ' <eos> ')
    txt = txt.replace('?', ' <eos> ')
    txt = txt.replace('!', ' <eos> ')
    txt = txt.replace('\n', ' <eos> ')
    txt = txt + ' <eos> '
    txt = re.sub('( <eos> )+', ' <eos> ', txt)

    # Replace all punctuation with special character
    txt = txt.translate(str.maketrans(punct, '_'*len(punct)))
    txt = txt.replace('_', ' <punct> ')

    # Collapse neighboring spaces
    txt = re.sub('[ ]+', ' ', txt, count = 0)
    
    return txt.strip()

def concat_docs(lang, train_dir):
    '''Concatenate all texts for a language.'''
    res = ''
    fns = list((train_dir/lang).glob('*.txt'))
    for fn in fns:
        res = res + fn.read_text()
    return res

def txt2list(txt, min_chars = 10):
    
    '''Preprocess text, and convert to list of sentences.'''
    
    res = preprocess(txt)
    res = res.strip().split('<eos>')
    
    # We also want to remove items consting only of whitespace and numbers
    res = list(map(lambda x: x.strip(), res))
    
    # Note for the minimum size: ' <eos> ' is already 7 long
    res = list(filter(lambda x: x != '' and x != '<num>' and len(x) >= min_chars, res))
    res = list(map(lambda x: x + ' <eos>', res))
    
    return res

def concat_random_sent(txt, p = 0.02):
    
    '''At random locations (probability p), concatenate two neighboring sentences.'''
    
    idxs = np.random.choice(range(len(txt) - 1), size = int(p*len(txt)), replace=False)
    
    for i in idxs:
        txt[i + 1] = txt[i] + ' ' + txt[i + 1]
        
    res = [x for i,x in enumerate(txt) if i not in idxs] 
    
    return res

def numericalize(X, word2idx, maxlen = 50):
    m = X.shape[0]
    res = np.ones((m, maxlen))
    for row in range(m):
        temp = np.array([word2idx[w] for w in X[row].split()])
        n = min(len(temp), maxlen)
        res[row, :n] = temp[:n]
    return res.astype(np.int32)

def de_numericalize(X, idx2word):
    res = []
    for row in range(X.shape[0]):
        res.append(' '.join([idx2word[i] for i in X[row][X[row] != 1]]))
    return res

def subsamp_disc_prob(idx_freq):
    '''Creates an index of discard probabilities for the skip-gram subsampling.'''
    const = 1e-5  # From Mikolov et al 2013
    rel_freq = idx_freq / (np.sum(idx_freq) - idx_freq[1])
    rel_freq[1] = 1  # We treat padding separately, otherwise it skews freq distribution
    disc_prob = 1 - np.sqrt(np.minimum(const / rel_freq, 1))
    return disc_prob

def negsamp_prob(idx_freq):
    const = 3/4  # From Mikolov et al 2013
    rel_freq = idx_freq / (np.sum(idx_freq))
    rel_freq[1] = 0.01  # Manually assign 1% prob to padding
    rel_freq = rel_freq ** const
    rel_freq = rel_freq / np.sum(rel_freq)
    return rel_freq

def skipgram_data(X, idx_freq, k = 5, as_tensor = True):
    
    m, n = X.shape

    # Create correct context-target pairs
    context_idx = np.random.randint(0, n, size = m)
    target_idx = np.random.randint(1, k, size = m)
    target_idx = (np.random.randint(0, 2, size = m)*2-1)*target_idx + context_idx
    target_idx = np.minimum(np.maximum(target_idx, 0), n-1)
    context = X[range(m), context_idx]
    target = X[range(m), target_idx]
    
    # Discard frequent words
    disc_prob = subsamp_disc_prob(idx_freq)
    keep_mask = (np.random.rand(m) > disc_prob[context])
    context = context[keep_mask]
    target = target[keep_mask]
    
    correct_num, fake_num, total  = (len(context), len(context)*5, len(context)*6)

    # Create incorrect pairs
    neg_samp_p = negsamp_prob(idx_freq)
    fake_target = np.random.choice(len(idx_freq), fake_num, p=neg_samp_p)
    fake_context = np.repeat(context, k)
    
    # Concatenate all
    context = np.concatenate((context, fake_context)).reshape((total, 1))
    target = np.concatenate((target, fake_target)).reshape((total, 1))
    X_res = np.concatenate((context, target), axis = 1)
    
    # Classification target
    y = np.concatenate((np.ones(correct_num), np.zeros(fake_num)))

    # Shuffle data
    shufl = np.random.permutation(total)
    X_res, y = (X_res[shufl], y[shufl])
    
    if as_tensor:   # Convert to pytorch tensor
        X_res = torch.from_numpy(X_res).type(torch.int64)
        y = torch.from_numpy(y).type(torch.float32)

    return (X_res, y)