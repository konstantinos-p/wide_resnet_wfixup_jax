from omegaconf import DictConfig
import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state
import numpy as np
import optax
import tensorflow_datasets as tfds
import tensorflow as tf
from typing import Any
import models
import hydra

ModuleDef = Any
KeyArray = random.KeyArray
Array = Any
DTypeLikeFloat = Any
DTypeLikeComplex = Any
DTypeLikeInexact = Any
RealNumeric = Any


def cross_entropy_loss(*, logits, labels):
    """
    The cross-entropy loss.
    Parameters
    ----------
    logits: float
        The prediction preactivations of the neural network.
    labels:
        The groundtruth labels.

    Returns
    -------
        : optax.softmax_cross_entropy
        The softmax_cross_entropy loss.

    """
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()


def compute_metrics(*, logits, labels):
    """
    Computes the crossentropy loss and the accuracy for a given set of predictions and groundtruth labels.
    Parameters
    ----------
    logits: float
        The prediction preactivations of the neural network.
    labels:
        The groundtruth labels.

    Returns
    -------
    metrics: dict
        A python dictionary with keys "loss" and "accuracy" corresponding to the cross-entropy loss and the accuracy
        for some logits and labels.

    """
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics


def get_datasets(cfg):
    """
    Load MNIST train and test datasets into memory.
    Parameters
    ----------
    cfg : DictConfig
        The configuration file for the experiment.

    Returns
    -------
    train_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to the training set.
    test_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to the test set.
    """
    if cfg.hyperparameters.dataset_dir == 'default':
        ds_builder = tfds.builder('Cifar10')
        ds_builder.download_and_prepare()
        train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
        test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    else:
        train_ds = tfds.load(name='Cifar10', data_dir=cfg.hyperparameters.dataset_dir, split='train', batch_size=-1)
        test_ds = tfds.load(name='Cifar10', data_dir=cfg.hyperparameters.dataset_dir, split='test', batch_size=-1)
    train_ds['image'] = jnp.float32(train_ds['image'])/255.
    test_ds['image'] = jnp.float32(test_ds['image'])/255.

    train_ds['label'] = jnp.int32(train_ds['label'])
    test_ds['label'] = jnp.int32(test_ds['label'])

    train_ds = {i: train_ds[i] for i in train_ds if i != 'id'}
    test_ds = {i: test_ds[i] for i in test_ds if i != 'id'}
    return train_ds, test_ds


def create_train_state(rng, cfg):
    """
    Creates initial `TrainState`.
    Parameters
    ----------
    rng : jax.random.PRNGKey
        Pseudo-random number generator (PRNG) key for the random initialization of the neural network.
    cfg : DictConfig
        The configuration file for the experiment.

    Returns
    -------
        :train_state.TrainState
        The initial training state of the experiment.
    """
    model_cls = getattr(models, cfg.hyperparameters.model)
    network = model_cls()
    params = network.init(rng, jnp.ones([1, 32, 32, 3]), train=False)
    tx = optax.sgd(cfg.hyperparameters.learning_rate, cfg.hyperparameters.momentum)
    return train_state.TrainState.create(apply_fn=network.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch, dropout_rng):
    """
    Trains the neural network for a single step."
    Parameters
    ----------
    state : train_state.TrainState
        The initial training state of the experiment.
    batch : dict
        Dictionary with keys 'image' and 'label' corresponding to a batch of the training set.
    dropout_rng : jax.random.PRNGKey
        Pseudo-random number generator (PRNG) key for the randomness of the dropout layers.

    Returns
    -------
    state : train_state.TrainState
        The training state of the experiment.
    metrics : dict
        A python dictionary with keys "loss" and "accuracy" corresponding to the cross-entropy loss and the accuracy
        for some logits and labels.
    new_dropout_rng : jax.random.PRNGKey
        New pseudo-random number generator (PRNG) key for the randomness of the dropout layers.

    """
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        logits = state.apply_fn(params, batch['image'], train=True, rngs={'dropout': dropout_rng})
        loss = cross_entropy_loss(logits=logits, labels=batch['label'])
        return loss, logits
    grad_fn = jax.grad(loss_fn, has_aux=True)
    grads, logits = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=batch['label'])
    return state, metrics, new_dropout_rng


@jax.jit
def eval_step(state, batch):
    """
    A single evaluation step of the output logits of the neural network for a batch of inputs, as well as the
    cross-entropy loss and the classification accuracy.
    Parameters
    ----------
    state : train_state.TrainState
        The training state of the experiment.
    batch : dict
        Dictionary with keys 'image' and 'label' corresponding to a batch of the training set.

    Returns
    -------
    metrics : dict
        A python dictionary with keys "loss" and "accuracy" corresponding to the cross-entropy loss and the accuracy
        for some logits and labels.
    """
    logits = state.apply_fn(state.params, batch['image'], train=False)
    return compute_metrics(logits=logits, labels=batch['label'])


def train_epoch(state, train_ds, batch_size, epoch, rng):
    """
    Train for a single epoch.
    Parameters
    ----------
    state : train_state.TrainState
        The training state of the experiment.
    train_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to the training set.
    batch_size : int
        The size of the batch.
    epoch : int
        The number of the current epoch.
    rng : jax.random.PRNGKey
        Pseudo-random number generator (PRNG) key for the random initialization of the neural network.

    Returns
    -------
    state : train_state.TrainState
        The new training state of the experiment.

    """
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch*batch_size]
    perms = perms.reshape((steps_per_epoch,batch_size))
    batch_metrics = []
    dropout_rng = jax.random.split(rng, jax.local_device_count())[0]
    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, metrics, dropout_rng = train_step(state, batch, dropout_rng)
        batch_metrics.append(metrics)

    #compute mean of metrics across each batch in epoch.train_state
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0] # jnp.mean does not work on lists
    }

    print('train epoch: %d, loss %.4f, accuracy: %.2f' % (epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy']*100))

    train_log_dir = 'logs/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    with train_summary_writer.as_default():
        tf.summary.scalar('accuracy', epoch_metrics_np['accuracy'], step=epoch)

    return state


def eval_model(state, test_ds):
    """

    Parameters
    ----------
    state : train_state.TrainState
        The training state of the experiment.
    test_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to the test set.

    Returns
    -------


    """
    test_ds_new = {i: test_ds[i] for i in test_ds if i != 'id'}
    metrics = eval_step(state, test_ds_new)
    metrics = jax.device_get(metrics)
    summary = jax.tree_util.tree_map(lambda x: x.item(), metrics)
    return summary['loss'], summary['accuracy']


@hydra.main(version_base=None, config_path="conf", config_name="fixupwideresnet_train")
def train_network(cfg : DictConfig):
    """

    Parameters
    ----------
    cfg : DictConfig
        The configuration file for the experiment.
    Returns
    -------
    test_accuracy : float
        The final test accuracy of the trained model. This is useful when doing hyperparameter search with optuna.

    """
    train_ds, test_ds = get_datasets(cfg)

    rng = jax.random.PRNGKey(cfg.hyperparameters.prngkeyseed)#0
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(init_rng,cfg)
    del init_rng #Must not be used anymore

    num_epochs = cfg.hyperparameters.epochs#10
    batch_size = cfg.hyperparameters.batch_size#32

    test_log_dir = 'logs/test'
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for epoch in range(1,num_epochs+1):
        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        state = train_epoch(state, train_ds, batch_size, epoch, input_rng)
        # Evaluate on the test set after each training epoch
        test_loss, test_accuracy = eval_model(state, test_ds)
        print('test epoch: %d, loss: %.2f, accuracy: %.2f' % (epoch, test_loss, test_accuracy*100))

        with test_summary_writer.as_default():
            tf.summary.scalar('accuracy', test_accuracy, step=epoch)

    return test_accuracy


if __name__ == '__main__':
    train_network()