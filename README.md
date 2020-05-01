# Sequential non-normal initializers for RNNs
PyTorch and Keras functions and classes implementing the non-normal initializers proposed in the following paper:

Orhan AE, Pitkow X (2020) [Improved memory in recurrent neural networks with sequential non-normal dynamics.](https://openreview.net/forum?id=ryx1wRNFvB) International Conference on Learning Representations (ICLR 2020).

The file [`NonnormalInit.py`](https://github.com/eminorhan/nonnormal-init/blob/master/NonnormalInit.py) contains plug-and-play Keras initializer classes implementing the proposed non-normal initializers for RNNs. The code was tested with `keras==2.2.4`, other versions may or may not work.

The files [`NonnormalInit_torchrnn.py`](https://github.com/eminorhan/nonnormal-init/blob/master/NonnormalInit_torchrnn.py) and [`NonnormalInit_torchlstm.py`](https://github.com/eminorhan/nonnormal-init/blob/master/NonnormalInit_torchlstm.py) contain torch functions implementing the proposed non-normal initializers for vanilla RNNs and LSTMs, respectively. The `ramp_init` function in [`NonnormalInit_torchlstm.py`](https://github.com/eminorhan/nonnormal-init/blob/master/NonnormalInit_torchlstm.py) implements the "mixed" initialization strategy discussed in section 3.3 of the paper. The code was tested with `torch==0.4.0`, other versions may or may not work.

The remaining files can be used to replicate the results in Figure 3. Please contact me for raw data from this figure (it was too large to upload here). An example usage would be as follows:

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

For the language modeling experiments, we used the Salesforce [awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm) repository, as described in sections 3.1.2 and 3.3 of the paper (with the torch initializers provided here: [`NonnormalInit_torchrnn.py`](https://github.com/eminorhan/nonnormal-init/blob/master/NonnormalInit_torchrnn.py) and [`NonnormalInit_torchlstm.py`](https://github.com/eminorhan/nonnormal-init/blob/master/NonnormalInit_torchlstm.py)). Again, please feel free to contact me for raw simulation results from these experiments as they were too large to upload here.
