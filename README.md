# PyTorch Implementation of Gated DRGT

This repo contains the code for our SIGIR 2019 paper: [Encoding Syntactic Dependency and Topical Information for Social Emotion Classification](https://dl.acm.org/doi/10.1145/3331184.3331287).

# Requirements

- python 3.5
- pytorch 0.3.1
- numpy
- pickle

# File Discription

- `parameter.py`: contains the parameters of Gated DRGT
- `model.py`: the code of the Gated DRGT model
- `run.py`: the code for training and testing Gated DRGT
- `dataset/`
  - `parser_data.pickle`: [SinaNews Dataset](https://ieeexplore.ieee.org/document/7904683). Each sample has been converted into a denpendency tree with [LTP](https://github.com/HIT-SCIR/ltp).
  - `doc_topics.npy`: the LDA topic distribution of each sample in the dataset.
  - `label.npy`: the label of each sample.
  - `w2v_model.pickle`: the pre-trained word2vec model.

# Data Preprocessing

Each document is cut into sentences, e.g., doc = [[sentence_1, sentence_2, ...]]

Each sentence is converted into a syntactic tree, e.g., sentence_1 = [(A, ROOT), [(B, SBV), [(C, VOB)]], [(D, ATT)]]


Each dependency relation is mapped to an ID with the following rules:

SBV:1, VOB:2, IOB:3, FOB:4, DBL:5, ATT:6, ADV:7, CMP:8, COO:9, POB:10, LAD:11, RAD:12, IS:13, WP:14, HED:15


# Acknowledgements

Some code is based on [TreeLSTM](https://github.com/Kailianghu/Tree-LSTM). Thanks for sharing the code.

# Cite
  
If you find the code helpful, please kindly cite the paper:
```
@inproceedings{10.1145/3331184.3331287,
author = {Wang, Chang and Wang, Bang and Xiang, Wei and Xu, Minghua},
title = {Encoding Syntactic Dependency and Topical Information for Social Emotion Classification},
year = {2019},
publisher = {Association for Computing Machinery},
doi = {10.1145/3331184.3331287},
booktitle = {Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {881–884},
keywords = {topic model, recursive neural network, social emotion classification, dependency embedding},
location = {Paris, France},
series = {SIGIR'19}
}
```