# Building a Language Detection Algorithm

This repo builds a language detection model that can be used for the 21 languages spoken in the European Union. For any given text, the model predicts its language.

This project is one of the challenge problem for the fellowship.ai application. The application organizers asked to build a model on the [EU Parliament Parallel Corpus](http://www.statmt.org/europarl/) a corpus of ~5 GB of text files (1.5 GB zipped). They also supplied the [test](https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/language-detection/europarl-test.zip) set to evaluate predictions on. 

This repo builds a model that achieves near-perfect test set accuracy. While fitting the model to the whole data takes a significant amount of computation time (XXX), the same model fitted to a 1% subsample of the training data also achieves very good accuracy (XXXX).

## Modelling Challanges

On the face of it, the project seems like a simple sequence classification task. Unless the test set contains text whose languages is ambiguous (the same text could pass as, say both Chezh and Slovekian) or contains extremely rare words, we would expect a good classifier to achieve near-perfect accuracy.

While near-perfect accuracy is indeed possible, there are two possible challenges that need to be overcome.

First, we can't rely on language specific information (after all, that's what we are trying to predict). This makes many standard NLP preprocessing steps unavailable. For example, we can't assign {'go', 'goes', 'going'} to the same word token. Even lower-casing is debatable, as some languages (for example German) capitalize words differently, a pattern that can be exploited to predict a language (German capitalizes all nouns).

Second, I chose not to use any pretrained models, or transfer learning. The challenge description wasn't clear if doing so is allowed or not; to err on the safe side, I only rely on the supplied training set. Normally, pretrained models and transfer learning is a great resource -- after all, the internet has practically unlimited text data one can pretrain a model on.

## Character or Word Level?

Should we build a model based on sequences f characters, or sequences f words? I expect both types of models to be able to achieve near-perfect classification results, so let's have a look at other considerations.

If our test set were to contain many words not appearing in the training set, a character-level model would be more appropriate. After all, a word-level model can't generalize outside the words it has seen. However, our training corpus is ~5 GB large, so we would expect it to contain the vast majority of common words in each language. 

A character level model also benefits from lower memory usage. After all, there are only a few hundred characters for all languages combined, but there are at least a few hundred thousand words. The embedding matrix of a word-level model would be enourmous.

On the flip-side, word level models come out ahead in computation time, both for training and for inference. A sentence that coontains 20 words will contain around 150 characters.

Moreover, a character level model needs to learn many more interactions, since words are more unique to a language than characters are. That means more hidden activations, and possibly a more complext architecture for a character level model.