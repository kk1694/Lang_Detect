# Building a Language Detection Algorithm

This repo builds a language detection model that can be used for the 21 languages spoken in the European Union. For any given text, the model predicts its language.

This project is one of the challenge problem for the fellowship.ai application. The application organizers asked to build a model on the [EU Parliament Parallel Corpus](http://www.statmt.org/europarl/) a corpus of ~5 GB of text files (1.5 GB zipped). They also supplied the [test](https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/language-detection/europarl-test.zip) set to evaluate predictions on. 

## Summary

The model trained on the whole dataset achieves near-perfect (99.9%+) test set accuracy. While fitting the model to the whole data takes a significant amount of computation time (~2 hours), the same model fitted to a 1% subsample of the training data also achieves very good results (98.7%).

The table below lists the model results based on sample size. 


| Dataset           | Model              | Accuracy           | Training Time      | Inference Time     | Vocabulary size    | Link              |
| ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| Full Dataset  | Word Level      | 99.96%             | 130 min      | 1.22 secs | 403619         | [here](https://github.com/kk1694/Lang_Detect/blob/master/Lang_Class.ipynb) |
| 10% Sample    | Word Level      | 99.90%             | 12  min      | 1.22 secs | 433309         | [here](https://github.com/kk1694/Lang_Detect/blob/master/Lang_Class_10pct.ipynb) |
| 1% Sample     | Word Level      | 98.79%             | 1   min      | 1.17 secs | 151746        | [here](https://github.com/kk1694/Lang_Detect/blob/master/Lang_Class_1pct.ipynb) |
| 1% Sample     | Character Level | 99.44%             | 7   min      | 1.79 secs  | 327            | [here](https://github.com/kk1694/Lang_Detect/blob/master/Lang_Class_charlvl_1pct.ipynb) |

Training time involves all preprocessing and model fitting (but not download time). Inference time involves predicting the ~20k sentences of the test set. The runs are done on a google cloud virtual machine with P100 GPU (the [main notebook](https://github.com/kk1694/Lang_Detect/blob/master/Lang_Class.ipynb) contains full specification).

## Model Description

The model is based on a recurrent neural net. After some very basic preprocessing, I embed words into a 50 dimensional  space. I feed the resulting embeddings through a standard GRU followed by a linear layer.

![Model Illustration](model_illustration.jpg)

The [main notebook](https://github.com/kk1694/Lang_Detect/blob/master/Lang_Class.ipynb) contains detailed steps; as well as justification behind the hyper-parameter choices.

## Description of Files

- [Lang_Class.ipynb](https://github.com/kk1694/Lang_Detect/blob/master/Lang_Class.ipynb): The main model, trained on the whole dataset. Contains explanations.
- [utils.py](https://github.com/kk1694/Lang_Detect/blob/master/utils.py): Contains a list of (relatively uninteresting) helper functions.
- [Download_Data.ipynb](https://github.com/kk1694/Lang_Detect/blob/master/Download_Data.ipynb): Downloads data, and puts in the appropriate directories.
- [Create_Smaller_Training_Set.ipynb](https://github.com/kk1694/Lang_Detect/blob/master/Create_Smaller_Training_Set.ipynb): Copies a random subset of the data to a new directory (for faster model building).
- [Lang_Class_10pct.ipynb](https://github.com/kk1694/Lang_Detect/blob/master/Lang_Class_10pct.ipynb): Trains the main model on 10% of the data. 
- [Lang_Class_10pct.ipynb](https://github.com/kk1694/Lang_Detect/blob/master/Lang_Class_1pct.ipynb): Trains the main model on 1% of the data.
- [Lang_Class_charlvl_1pct.ipynb](https://github.com/kk1694/Lang_Detect/blob/master/Lang_Class_charlvl_1pct.ipynb): Trains a character-level model on 1% of the data.