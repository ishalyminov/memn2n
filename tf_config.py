import tensorflow as tf


def configure(in_config):
    tf.flags.DEFINE_float(
        'learning_rate',
        in_config['learning_rate'],
        'Learning rate for Adam Optimizer'
    )
    tf.flags.DEFINE_float(
        'epsilon',
        in_config['epsilon'],
        'Epsilon value for Adam Optimizer'
    )
    tf.flags.DEFINE_float(
        'max_grad_norm',
        in_config['max_grad_norm'],
        'Clip gradients to this norm')
    tf.flags.DEFINE_integer(
        'evaluation_interval',
        in_config['evaluation_interval'],
        "Evaluate and print results every x epochs"
    )
    tf.flags.DEFINE_integer(
        'batch_size',
        in_config['batch_size'],
        'Batch size for training'
    )
    tf.flags.DEFINE_integer(
        'hops',
        in_config['hops'],
        'Number of hops in the Memory Network'
    )
    tf.flags.DEFINE_integer(
        'epochs',
        in_config['epochs'],
        'Number of epochs to train for'
    )
    tf.flags.DEFINE_integer(
        'embedding_size',
        in_config['embedding_size'],
        'Embedding size for embedding matrices'
    )
    tf.flags.DEFINE_integer(
        'memory_size',
        in_config['memory_size'],
        'Maximum size of memory'
    )
    tf.flags.DEFINE_integer(
        'task_id',
        in_config['task_id'],
        "bAbI task id, 1 <= id <= 6"
    )
    tf.flags.DEFINE_integer(
        'random_state',
        in_config['random_state'],
        'Random state'
    )