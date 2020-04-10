import numpy as np
import optimizers
import pickle


class Solver(object):
    """A solver encapsulates all the logic necessary for training classification models. The solver performs stochastic
    gradient descent using different update rules defined in optimizer.
    """
    def __init__(self, model, data, **kwargs):
        """
        Required args:
            model:
            data:

        Optional args:
            update_rule: A string giving the name of an update rule in optimizer.py, default is sgd
            optim_config: A dictionary containing hyperparameters that will be passed to the choosen update rule.
            lr_decay:
            batch_size:
            num_epochs:
            print_every:
            verbose:
            num_train_samples:
            num_val_samples:
            checkpoint_name:
        """
        self.model = model
        self.x_train = data['X_train']
        self.y_train = data['y_train']
        self.x_val = data['X_val']
        self.y_val = data['y_val']

        # Unpack keyword arguments
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.num_train_samples = kwargs.pop('num_train_samples', 1000)
        self.num_val_samples = kwargs.pop('num_val_samples', None)
        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            raise ValueError('Unrecognized arguments')

        # Make sure the update rule exists within the imported package, then replace the string name with actual function
        if not hasattr(optimizers, self.update_rule):
            raise ValueError("Invalid update rule: %s" % self.update_rule)

        self.update_rule = getattr(optimizers, self.update_rule)
        self._reset()

    def _reset(self):
        """Setup book-keeping variables for optimization
        """
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter NOTE: BUT WHY?
        self.optim_configs = {}
        for param in self.model.params:
            config = {}
            for key, val in self.optim_config.items():
                config[key] = val
            self.optim_configs[param] = config

    def _step(self):
        """Make a single gradient update
        """
        num_train = self.x_train.shape[0]

        # Create a mini-batch of training data
        batch_mask = np.random.choice(num_train, self.batch_size)
        x_batch = self.x_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Compute loss and graident
        loss, grads = self.model.loss(x_batch, y_batch)
        self.loss_history.append(loss)

        # Performs parameter update
        for param, weight in self.model.params.items():
            grad_w = grads[param]
            config = self.optim_configs[param]
            next_w, next_config = self.update_rule(weight, grad_w, config)

            self.model.params[param] = next_w
            self.optim_configs[param] = next_config

    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return

        checkpoint = {
            'model': self.model,
            'update_rule': self.update_rule,
            'lr_decay': self.lr_decay,
            'optim_config': self.optim_config,
            'batch_size': self.batch_size,
            'num_train_samples': self.num_train_samples,
            'num_val_samples': self.num_val_samples,
            'epoch': self.epoch,
            'loss_history': self.loss_history,
            'train_acc_history': self.train_acc_history,
            'val_acc_history': self.val_acc_history,
        }

        filename = "%s_epoch_%s.pkl" % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print "Saving checkpoint to %s" % filename

        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

    def check_accuracy(self, x, y, num_samples=None, batch_size=100):
        """Check accuracy of the model on the provided data
        """
        # Subsample the data if num_samples is provided
        N = x.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            x = x[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1

        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(x[start:end])
            y_pred.append(np.argmax(scores, axis=1))

        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        """Run optimization to train the model
        """
        num_train = self.x_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Print the training loss if verbose is enabled
            if self.verbose and t % self.print_every == 0:
                print "(Iteration %d / %d) loss: %f" % (t + 1, num_iterations, self.loss_history[-1])

            # At every end of epoch, update the optim config before we pass it into update_rule in next iteration of step()
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for param in self.optim_configs:
                    self.optim_configs[param]['learning_rate'] *= self.lr_decay

            # Check training and val accuracy on the first iteration, the last iteration, and at end of each epoch.
            first_itr = (t == 0)
            last_itr = (t == num_iterations - 1)
            if first_itr or last_itr or epoch_end:
                train_acc = self.check_accuracy(self.x_train, self.y_train, num_samples=self.num_train_samples)
                val_acc = self.check_accuracy(self.x_val, self.y_val, num_samples=self.num_val_samples)

                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()

                if self.verbose:
                    print "(Epoch %d / %d) train acc: %f; val_acc: %f" % (self.epoch, self.num_epochs, train_acc, val_acc)

                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for key, val in self.model.params.items():
                        self.best_params[key] = val.copy()

        # At the end of training, swap the best params into the model
        self.model.params = self.best_params
