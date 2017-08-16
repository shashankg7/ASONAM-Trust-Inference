Codebase (partial) for ASONAM'17 paper [Simultaneous Inference of User Representations and Trust](https://arxiv.org/abs/1706.00923).

#### Main file :
train1.py

#### Model file :
model1.py

#### Data Handler and Segmentation file:
data_handler.py

#### To run the program
python train1.py num_items_threshold   degree_threshold  segment_using_degree_or_items

ex:

python train1.py 100 50 degree/items

#### Dependencies :

1. Theano
2. Numpy
3. Networkx

### NOTE
Please note that this repo. is still incomplete, in the sense that experiments cannot be reproduced yet. It has model file and some data pre-processing scripts.

I am planning to update it will fully functional repo, where interested users can reproduce all the experiments.
