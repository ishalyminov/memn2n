"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

import random
from itertools import chain
from six.moves import range, reduce
import logging

from sklearn import metrics
import tensorflow as tf
import numpy as np

from dialog_data_utils import (
    vectorize_data_dialog,
    get_candidates_list,
    load_task,
    vectorize_answers
)
from memn2n import MemN2N

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__file__)

tf.flags.DEFINE_float(
    "learning_rate",
    0.01,
    "Learning rate for Adam Optimizer."
)
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer(
    "evaluation_interval",
    1,
    "Evaluate and print results every x epochs"
)
tf.flags.DEFINE_integer("batch_size", 8, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer(
    "embedding_size",
    20,
    "Embedding size for embedding matrices."
)
tf.flags.DEFINE_integer("memory_size", 64, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 6")
tf.flags.DEFINE_integer("random_state", 273, "Random state.")
tf.flags.DEFINE_string(
    "data_dir",
    "../babi_tools/dialog-bAbI-tasks/",
    "Directory containing bAbI tasks"
)
tf.flags.DEFINE_string(
    "data_dir_plus",
    "../babi_tools/dialog-bAbI-tasks/",  # "../babi_tools/babi_plus",
    "Directory containing bAbI+ tasks"
)
FLAGS = tf.flags.FLAGS

random.seed(FLAGS.random_state)

print("Started Task:", FLAGS.task_id)

# task data
train_babi, dev_babi, test_babi, test_oov_babi = load_task(FLAGS.data_dir, FLAGS.task_id)
train_plus, dev_plus, test_plus, test_oov_plus = load_task(FLAGS.data_dir_plus, FLAGS.task_id)

all_dialogues_babi = train_babi + dev_babi + test_babi + test_oov_babi
all_dialogues_babi_plus = train_plus + dev_plus + test_plus + test_oov_plus

data = reduce(
    lambda x, y: x + y,
    all_dialogues_babi + all_dialogues_babi_plus,
    []
)
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


def train_model(in_model, in_train_sqa, in_test_sqa, in_batches):
    best_train_accuracy, best_test_accuracy = 0.0, 0.0

    for t in range(1, FLAGS.epochs+1):
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

        if t % FLAGS.evaluation_interval == 0:
            # evaluate on the whole trainset
            train_preds = in_model.predict(s_train, q_train)
            train_acc = metrics.accuracy_score(
                train_preds,
                train_labels
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
    dialogues_train = map(lambda x: [x[-1]], train_babi)
    dialogues_test = map(lambda x: [x[-1]], test_babi)

    data_train = reduce(lambda x, y: x + y, dialogues_train, [])
    data_test = reduce(lambda x, y: x + y, dialogues_test, [])

    train_s, train_q, train_a = vectorize_data_dialog(
        data_train,
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
    print("Testing Size (dialogues)", len(dialogues_test))
    print("Training Size (stories)", len(data_train))
    print("Testing Size (stories)", len(data_test))

    tf.set_random_seed(FLAGS.random_state)
    batch_size = FLAGS.batch_size
    optimizer = tf.train.AdamOptimizer(
        learning_rate=FLAGS.learning_rate,
        epsilon=FLAGS.epsilon
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
            (test_s, test_q, test_a),
            batches
        )
    return best_accuracy_per_epoch


if __name__ == '__main__':

    accuracies = main()
    print ('train: {0:.3f}, test: {1:.3f}'.format(*accuracies))
