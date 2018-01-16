"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

import random
from argparse import ArgumentParser
from itertools import chain
from six.moves import range, reduce
import logging
import sys
from os import path
import json

from sklearn import metrics
import tensorflow as tf
import numpy as np

from dialog_data_utils import (
    vectorize_data_dialog,
    get_candidates_list,
    load_task,
    vectorize_answers
)
from tf_config import configure
from memn2n import MemN2N

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__file__)


def configure_argument_parser():
    parser = ArgumentParser(description='train MemN2N on bAbI/bAbI+ dialogs')
    parser.add_argument('train_dialogs', help='train dialogs root')
    parser.add_argument('test_dialogs', help='test dialogs root')
    parser.add_argument(
        '--predict_last_turn_only',
        default=False,
        action='store_true',
        help='whether to only test on the last (API) turns'
    )
    parser.add_argument(
        '--ignore_api_calls',
        default=False,
        action='store_true',
        help='whether to ignore API calls while loading data'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='babi_plus_dialog_single.json',
        help='MemN2N config'
    )

    return parser

parser = configure_argument_parser()
args = parser.parse_args()

CONFIG_FILE = path.join(path.dirname(__file__), args.config)
with open(CONFIG_FILE) as config_in:
    CONFIG = json.load(config_in)
configure(CONFIG)
tf.flags.DEFINE_string(
    "data_dir",
    args.train_dialogs,
    "Directory containing bAbI tasks"
)
tf.flags.DEFINE_string(
    "data_dir_plus",
    args.test_dialogs,
    "Directory containing bAbI+ tasks"
)
FLAGS = tf.flags.FLAGS
print('{}:\t{}'.format('data_dir', FLAGS.data_dir))
print('{}:\t{}'.format('data_dir_plus', FLAGS.data_dir_plus))
for key, value in CONFIG.iteritems():
    print('{}:\t{}'.format(key, value))

random.seed(FLAGS.random_state)
np.random.seed(FLAGS.random_state)

print("Started Task:", FLAGS.task_id)

# task data
train_babi, dev_babi, test_babi, test_oov_babi = load_task(
    FLAGS.data_dir,
    FLAGS.task_id,
    args.ignore_api_calls
)
train_plus, dev_plus, test_plus, test_oov_plus = load_task(
    FLAGS.data_dir_plus,
    FLAGS.task_id,
    args.ignore_api_calls
)

all_dialogues_babi = train_babi + dev_babi + test_babi + test_oov_babi
all_dialogues_babi_plus = train_plus + dev_plus + test_plus + test_oov_plus

data = []
for dialogue in all_dialogues_babi + all_dialogues_babi_plus:
    data += dialogue

# data = reduce(
#     lambda x, y: x + y,
#     all_dialogues_babi + all_dialogues_babi_plus,
#     []
# )
max_story_size = max(map(len, (s for s, _, _ in data)))
mean_story_size = int(np.mean([len(s) for s, _, _ in data]))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data))) + 2
query_size = max(map(len, (q for _, q, _ in data)))
memory_size = min(FLAGS.memory_size, max_story_size)

answer_candidates = get_candidates_list(FLAGS.data_dir)

vocab = reduce(
    lambda x, y: x | y,
    (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)
)
vocab |= reduce(
    lambda x, y: x | y,
    [set(answer.split()) for answer in answer_candidates]
)
vocab = sorted(vocab)

word_idx = {c: i + 1 for i, c in enumerate(vocab)}
answer_idx = {
    candidate: i + 1
    for i, candidate in enumerate(answer_candidates)
}

vocab_size = len(word_idx) + 1  # +1 for nil word
answer_vocab_size = len(answer_idx) + 1
sentence_size = max(query_size, sentence_size)  # for the position

answers_vectorized = vectorize_answers(answer_candidates, word_idx, sentence_size)

print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)


# in_train_sqa - trainset
# in_train_eval_sqa - trainset for evaluation (may be API calls only)
# # in_test_sqa - testset for evaluation
def train_model(in_model, in_train_sqa, in_train_eval_sqa, in_test_sqa, in_batches):
    best_train_accuracy, best_test_accuracy = 0.0, 0.0

    for t in range(1, FLAGS.epochs+1):
        s_train, q_train, a_train = in_train_sqa
        s_train_eval, q_train_eval, a_train_eval = in_train_eval_sqa
        s_test, q_test, a_test = in_test_sqa
        train_labels = np.argmax(a_train, axis=1)
        train_eval_labels = np.argmax(a_train_eval, axis=1)
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

        if t % FLAGS.evaluation_interval == 0:
            # evaluate on the whole trainset
            eval_batch_size = 100
            train_preds = np.zeros(shape=train_eval_labels.shape, dtype=np.int32) 
            for batch_start in xrange(0, len(s_train_eval), eval_batch_size):
                batch_end = (batch_start + eval_batch_size) % (len(s_train_eval) + 1)
                preds = in_model.predict(s_train_eval[batch_start:batch_end], q_train_eval[batch_start:batch_end])
                train_preds[batch_start:batch_end] = preds
            train_acc = metrics.accuracy_score(
                train_preds,
                train_eval_labels
            )

            # evaluating on the whole testset
            test_preds = in_model.predict(s_test, q_test)
            test_acc = metrics.accuracy_score(
                test_preds,
                test_labels
            )

            logger.info('-----------------------')
            logger.info('Epoch:\t{}'.format(t))
            logger.info('Total Cost:\t{}'.format(total_cost))
            logger.info('Training Accuracy:\t{}'.format(train_acc))
            logger.info('Testing Accuracy:\t{}'.format(test_acc))
            logger.info('-----------------------')
            if best_test_accuracy < test_acc:
                best_train_accuracy, best_test_accuracy = train_acc, test_acc
    return best_train_accuracy, best_test_accuracy


def main():
    dialogues_train = map(lambda x: x, train_babi)
    dialogues_train_eval = map(lambda x: [x[-1]], train_babi) \
        if args.predict_last_turn_only \
        else map(lambda x: x, train_babi)

    dialogues_test = map(lambda x: [x[-1]], test_plus) \
        if args.predict_last_turn_only \
        else map(lambda x: x, test_plus)

    data_train = reduce(lambda x, y: x + y, dialogues_train, [])
    data_train_eval = reduce(lambda x, y: x + y, dialogues_train_eval, [])
    data_test = reduce(lambda x, y: x + y, dialogues_test, [])

    train_s, train_q, train_a = vectorize_data_dialog(
        data_train,
        word_idx,
        answer_idx,
        sentence_size,
        memory_size
    )
    train_eval_s, train_eval_q, train_eval_a = vectorize_data_dialog(
        data_train_eval,
        word_idx,
        answer_idx,
        sentence_size,
        memory_size
    )
    test_s, test_q, test_a = vectorize_data_dialog(
        data_test,
        word_idx,
        answer_idx,
        sentence_size,
        memory_size
    )

    print("Training Size (dialogues)", len(dialogues_train))
    print("Training/Evaluation Size (dialogues)", len(dialogues_train_eval))
    print("Testing Size (dialogues)", len(dialogues_test))
    print("Training Size (stories)", len(data_train))
    print("Training/Evaluation Size (stories)", len(data_train_eval))
    print("Testing Size (stories)", len(data_test))

    tf.set_random_seed(FLAGS.random_state)
    batch_size = FLAGS.batch_size
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=FLAGS.learning_rate
    )

    batches = zip(
        range(0, len(data_train) - batch_size, batch_size),
        range(batch_size, len(data_train), batch_size)
    )
    batches = [(start, end) for start, end in batches]


    with tf.Session() as sess:
        model = MemN2N(
            batch_size,
            vocab_size,
            sentence_size,
            memory_size,
            FLAGS.embedding_size,
            answers_vectorized,
            session=sess,
            hops=FLAGS.hops,
            max_grad_norm=FLAGS.max_grad_norm,
            optimizer=optimizer
        )
        best_accuracy_per_epoch = train_model(
            model,
            (train_s, train_q, train_a),
            (train_eval_s, train_eval_q, train_eval_a),
            (test_s, test_q, test_a),
            batches
        )
    return best_accuracy_per_epoch


if __name__ == '__main__':
    accuracies = main()
    print ('train: {0:.3f}, test: {1:.3f}'.format(*accuracies))
