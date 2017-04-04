"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

import json
import random
from optparse import OptionParser

from six.moves import range, reduce
import logging

from sklearn import metrics
import tensorflow as tf
import numpy as np

from ds_dialog_data_utils import load_task_ds, vectorize_data_dialog_ds
from dialog_data_utils import load_task, get_candidates_list
from memn2n import MemN2N

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__file__)

CONFIG_PATH = 'dialog_single.json'
with open(CONFIG_PATH) as config_in:
    CONFIG = json.load(config_in)

random.seed(CONFIG['random_state'])


def load_data(in_data_root, in_task_id):
    # task data
    train, dev, test, oov = load_task_ds(in_data_root, in_task_id)
    train_raw, dev_raw, test_raw, oov_raw = load_task(in_data_root, in_task_id)
    all_dialogues = train + dev + test + oov
    data = reduce(lambda x, y: x + y, all_dialogues, [])

    answer_candidates = get_candidates_list(in_data_root)
    answer_idx = dict(
        (candidate, i + 1)
        for i, candidate in enumerate(answer_candidates)
    )
    return {
        'ds': [train, dev, test, oov],
        'raw': [train_raw, dev_raw, test_raw, oov_raw],
        'answer_idx': answer_idx
    }


def train_model(in_model, in_train_sqa, in_test_sqa, in_batches):
    best_train_accuracy, best_test_accuracy = 0.0, 0.0
    for t in range(1, CONFIG['epochs'] + 1):
        s_train, q_train, a_train = in_train_sqa
        s_test, q_test, a_test = in_test_sqa
        train_labels = np.argmax(a_train, axis=1)
        test_labels = np.argmax(a_test, axis=1)
        np.random.shuffle(in_batches)
        total_cost = 0.0
        for start, end in in_batches:
            s = s_train[start:end]
            q = q_train[start:end]
            a = a_train[start:end]
            # back-propagating each batch
            cost_t = in_model.batch_fit(s, q, a)
            total_cost += cost_t

        if t % CONFIG['evaluation_interval'] == 0:
            # evaluate on the whole trainset
            train_preds = in_model.predict(s_train, q_train)
            train_acc = metrics.accuracy_score(train_preds, train_labels)

            # evaluating on the whole testset
            test_preds = in_model.predict(s_test, q_test)
            test_acc = metrics.accuracy_score(test_preds, test_labels)

            logger.info('-----------------------')
            logger.info('Epoch:\t{}'.format(t))
            logger.info('Total Cost:\t{}'.format(total_cost))
            logger.info('Training Accuracy:\t{}'.format(train_acc))
            logger.info('Testing Accuracy:\t{}'.format(test_acc))
            logger.info('-----------------------')
            best_train_accuracy, best_test_accuracy = max(
                (best_train_accuracy, best_test_accuracy),
                (train_acc, test_acc)
            )
    return best_train_accuracy, best_test_accuracy


def configure_option_parser():
    parser = OptionParser()

    parser.add_option(
        "--use_api_calls",
        dest="use_api_calls",
        help="whether to process API call turns",
        default=True
    )
    return parser


def main(in_data_root):
    print("Started Task:", CONFIG['task_id'])
    data_json = load_data(in_data_root, CONFIG['task_id'])

    # both DS and raw text features used
    train, dev, test, oov = data_json['ds']
    data_train = reduce(lambda x, y: x + y, train, [])
    data_test = reduce(lambda x, y: x + y, test, [])
    train_raw, dev_raw, test_raw, oov_raw = data_json['raw']
    data_train_raw = reduce(lambda x, y: x + y, train, [])
    data_test_raw = reduce(lambda x, y: x + y, test, [])

    answer_idx = data_json['answer_idx']

    data_train = reduce(lambda x, y: x + y, train, [])
    data_test = reduce(lambda x, y: x + y, test, [])

    train_s, train_q, train_a = vectorize_data_dialog_ds(
        data_train_raw,
        data_train,
        answer_idx,
        CONFIG['memory_size']
    )
    test_s, test_q, test_a = vectorize_data_dialog_ds(
        data_test_raw,
        data_test,
        answer_idx,
        CONFIG['memory_size']
    )

    print("Training Size (dialogues)", len(train))
    print("Testing Size (dialogues)", len(test))
    print("Training Size (stories)", len(data_train))
    print("Testing Size (stories)", len(data_test))

    tf.set_random_seed(CONFIG['random_state'])
    batch_size = CONFIG['batch_size']
    optimizer = tf.train.AdamOptimizer(
        learning_rate=CONFIG['learning_rate'],
        epsilon=CONFIG['epsilon']
    )

    batches = zip(
        range(0, len(data_train) - batch_size, batch_size),
        range(batch_size, len(data_train), batch_size)
    )
    batches = [(start, end) for start, end in batches]

    with tf.Session() as sess:
        model = MemN2N(
            CONFIG['batch_size'],
            2,
            len(data_train[0]),
            CONFIG['batch_size'],
            CONFIG['embedding_size'],
            answer_vocab_size=2,
            session=sess,
            hops=CONFIG['hops'],
            max_grad_norm=CONFIG['max_grad_norm'],
            optimizer=optimizer
        )
        best_accuracy_per_epoch = train_model(
            model,
            (train_s, train_q, train_a),
            (test_s, test_q, test_a),
            batches
        )
    return best_accuracy_per_epoch


if __name__ == '__main__':
    accuracies = main()
    print ('train: {0:.3f}, test: {1:.3f}'.format(*accuracies))
