a neural net evaluation function for playing chess 

paper - arXiv:1711.09667  
https://arxiv.org/pdf/1711.09667.pdf

architecture:-
pos2vec (dbn) -> 733-100-100-100 (unsupervised pre-training for feature extraction)
deep_chess network -> (2xdbn-100-100-2) all fc layers, supervised training

inputs to the network are encoded bitboards flattened to a vector of len 773
output is a softmax-ed prediction of a win-loss prediction probabilities 


========================TODO=================
learn about dbns,
learn about min-max, alpha-beta pruning
construct trainable tensor dataset of board states and resulting scores
train dbn
train deepchess
build playable model with minmax and nn_eval
implement svg server 
