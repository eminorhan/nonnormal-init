'''Tasks for RNNs'''

import generate_data

class Task(object):
    def __init__(self):
        pass

    def update(self, new_config):
        self.__dict__.update(new_config.__dict__)

    def __str__(self):
        return str(self.__dict__)

class AdditionTask(Task):
    def __init__(self):
        super(AdditionTask, self).__init__()
        self.n_train = 60000
        self.n_test = 10000
        self.T = 750
        self.num_classes = 1
        self.return_sequences = False
        self.hidden_activation = 'relu'
        self.output_activation = 'linear'
        self.loss_fnc = 'mean_squared_error'
        self.metrics = 'accuracy'  # dummy

    def generate_task_data(self):
        return generate_data.addition_task(self.n_train, self.n_test, self.T)

class CopyTask(Task):
    def __init__(self):
        super(CopyTask, self).__init__()
        self.n_train = 60000
        self.n_test = 10000
        self.T = 500
        self.num_classes = 9
        self.return_sequences = True
        self.hidden_activation = 'elu'
        self.output_activation = 'softmax'
        self.loss_fnc = 'categorical_crossentropy'
        self.metrics = 'accuracy'

    def generate_task_data(self):
        return generate_data.copy_task(self.n_train, self.n_test, self.T)

class psMNISTTask(Task):
    def __init__(self):
        super(psMNISTTask, self).__init__()
        self.num_classes = 10
        self.return_sequences=False
        self.hidden_activation = 'elu'
        self.output_activation = 'softmax'
        self.loss_fnc = 'categorical_crossentropy'
        self.metrics = 'accuracy'

    def generate_task_data(self):
        return generate_data.permuted_mnist_task()

def TaskLoader(task):
    if task=='addition':
        return AdditionTask()
    elif task=='copy':
        return CopyTask()
    elif task=='psmnist':
        return psMNISTTask()