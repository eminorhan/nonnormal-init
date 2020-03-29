# Sequential non-normal initializers for RNNs
PyTorch and Keras functions and classes implementing the non-normal initializers proposed in the following paper:

Orhan AE, Pitkow X (2019) [Improved memory in recurrent neural networks with sequential non-normal dynamics.](https://arxiv.org/abs/1905.13715) arXiv:1905.13715.

The file [`NonnormalInit.py`](https://github.com/eminorhan/nonnormal-init/blob/master/NonnormalInit.py) contains plug-and-play Keras initializer classes implementing the proposed non-normal initializers for RNNs. 

The files [`NonnormalInit_torchrnn.py`](https://github.com/eminorhan/nonnormal-init/blob/master/NonnormalInit_torchrnn.py) and [`NonnormalInit_torchlstm.py`](https://github.com/eminorhan/nonnormal-init/blob/master/NonnormalInit_torchlstm.py) contain torch functions implementing the proposed non-normal initializers for vanilla RNNs and LSTMs, respectively. The `ramp_init` function in [`NonnormalInit_torchlstm.py`](https://github.com/eminorhan/nonnormal-init/blob/master/NonnormalInit_torchlstm.py) implements the "mixed" initialization strategy discussed in section 2.5 of the paper.

The remaining files can be used to replicate the results in Figure 3. An example usage would be as follows:

```
python train.py --task 'copy' --init 'chain' --init_scale 1.02 --lr 5e-5 --rand_seed 3
```
where:

* `task` is the task (`copy, addition, psmnist`) 
* `init` is the initializer for the RNN (`chain, fbchain, orthogonal, identity`) 
* `init_scale` is the gain of the initializer (in the paper, this corresponds to `alpha` for the `chain` initializer, `beta` for the `fbchain` initializer, and `lambda` for the `orthogonal` and `identity` initializers) 
* `lr` is the learning rate for the rmsprop algorithm
* `rand_seed` is the random seed. 

See [`train.py`](https://github.com/eminorhan/nonnormal-init/blob/master/train.py) for more options.
