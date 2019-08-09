'''Main training script for running common RNN benchmarks'''

import os
import sys
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Activation
from keras.optimizers import rmsprop
from utils import InitLoader, EarlyStoppingByLossVal
from tasks import TaskLoader

parser = argparse.ArgumentParser(description='Non-normal initializers for vanilla RNNs.')
parser.add_argument('--task', type=str, default='copy', help='task (copy, addition, psmnist)')
parser.add_argument('--init', type=str, default='chain', help='initializer (chain, fbchain, orthogonal, identity)')
parser.add_argument('--init_scale', type=float, default=1.02, help='scale for initializer')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--rand_seed', type=int, default=1, help='random seed')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--hidden_units', type=int, default=100, help='number of hidden units')
parser.add_argument('--clip_norm', type=float, default=10.0, help='gradient norm clip')

args = parser.parse_args()

np.random.seed(args.rand_seed)

# load task
task = TaskLoader(args.task)

# load data
(x_train, y_train), (x_test, y_test) = task.generate_task_data()

optimizer = rmsprop(lr=args.lr, clipnorm=args.clip_norm)

# define initializers
kernel_initializer, recurrent_initializer = InitLoader(args.init, args.init_scale, args.hidden_units)

# define model
model = Sequential()
model.add(SimpleRNN(args.hidden_units,
                    kernel_initializer=kernel_initializer,
                    recurrent_initializer=recurrent_initializer,
                    activation=task.hidden_activation,
                    return_sequences=task.return_sequences,
                    input_shape=x_train.shape[1:]))
model.add(Dense(task.num_classes))
model.add(Activation(task.output_activation))
model.compile(loss=task.loss_fnc, optimizer=optimizer, metrics=[task.metrics])

# train model
callbacks = [EarlyStoppingByLossVal(monitor='val_loss', value=0.0025, verbose=0)]
hist = model.fit(x_train, y_train,
                 batch_size=args.batch_size,
                 epochs=args.epochs,
                 verbose=2,
                 validation_data=(x_test, y_test),
                 callbacks=callbacks)

# test model
scores = model.evaluate(x_test, y_test, verbose=0)

print('RNN test loss:', scores[0])
print('RNN test accuracy:', scores[1])

# save the results
model_id = args.task + '_' + args.init + '_' + '%.2f' % args.init_scale + '_' + '%.6f' % args.lr + '_' + '%i' % args.rand_seed
model.save(model_id + '.h5')
np.savez(model_id + '.npz', hist_acc=hist.history['acc'], hist_loss=hist.history['loss'], hist_val_acc=hist.history['val_acc'], 
         hist_val_loss=hist.history['val_loss'])
