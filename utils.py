import re
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

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
    errors = 0  # Keep track of how many files we couldn't open
    fns = list((train_dir/lang).glob('*.txt'))
    for fn in fns:
        try:
            res = res + fn.read_text()
        except:
            errors += 1
    if errors > 0:
        print(f'\n{errors} files not loaded for {lang}')
    return res

def txt2list(txt, min_chars = 3):
    
    '''Converts text to a list of sentences.'''
    
    res = txt.strip().split('<eos>')
    res = list(map(lambda x: x.strip(), res))
    
    # We also want to remove items consting only of whitespace,
    # consiting only of numbers, or less having less than min_char characters.   
    res = list(filter(lambda x: x != '' and x != '<num>' and len(x) >= min_chars, res))
    res = list(map(lambda x: x + ' <eos>', res))
    
    return res

def concat_random_sent(txt, p = 0.02):
    
    '''At random locations (probability p), concatenate two neighboring sentences.'''
    
    idxs = np.random.choice(range(len(txt) - 1), size = int(p*len(txt)), replace=False)
    
    for i in idxs:
        txt[i + 1] = txt[i] + ' ' + txt[i + 1]  
        
    # Delete first sentence; otherwise there would be duplicates
    res = [x for i,x in enumerate(txt) if i not in idxs] 
    
    return res

def numericalize(X, word2idx, maxlen = 32, pad = '<pad>'):
    
    '''Converts a (m,) array of sentences into a (m, maxlen) array of word indices.'''    
    
    m = X.shape[0]
    pad_idx = word2idx[pad]  # Should be equal to 1.
    res = np.ones((m, maxlen)) * pad_idx  # Array with only padding
    
    for row in tqdm(range(m), position = 0, leave = False):
        temp = np.array([word2idx[w] for w in X[row].split()])
        n = min(len(temp), maxlen)  # Truncate sentence at maxlen words.
        res[row, :n] = temp[:n]
        
    return res.astype(np.int32)

def de_numericalize(X, idx2word):
    
    '''Converts an array of word indices into a list of sentences.'''
    
    res = []
    for row in range(X.shape[0]):
        res.append(' '.join([idx2word[i] for i in X[row][X[row] != 1]]))
    return res

def exp_smooth(x, beta, bias_correct = True):
    
    '''Applies exponential smoothing to x.'''
    
    if beta == 0:
        return x
    
    m = x.shape[0]
    res = np.zeros(m)
    res[0] = (1 - beta)*x[0]
    
    for i in range(1, m):
        res[i] = beta*res[i-1] + (1 - beta)*x[i]
        
    if bias_correct:
        # Corrects for bias towards 0 in the first few elements
        corr = 1 - beta ** (np.arange(m) + 1)
        assert corr.shape == res.shape
        res = res / corr
        
    return res

def conv2np(tens):
    
    '''If given a tensor, converts to numpy, and takes object off the gpu.'''
    
    if torch.is_tensor(tens):
        if tens.is_cuda:
            return tens.detach().cpu().numpy()
        else:
            return tens.detach().numpy()
    else:
        return tens

def accuracy(pred, y):
    
    '''Returns accuracy of predictions (how often true label matches predicted label).'''
    
    pred = conv2np(pred); y = conv2np(y)  # Convert to numpy
    m = y.shape[0]
    
    assert pred.shape == (m,)  # Make sure both objects are rank 1 arrays of same shape
    assert y.shape == (m,)
    
    return np.sum(pred == y) / m