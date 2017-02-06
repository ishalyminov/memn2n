"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from itertools import chain
from six.moves import range, reduce
import logging
import sys

from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
import tensorflow as tf
import numpy as np

from dialog_data_utils import (
    load_task,
    vectorize_data_dialog,
    get_candidates_list
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
tf.flags.DEFINE_integer("batch_size", 1, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
tf.flags.DEFINE_integer(
    "embedding_size",
    20,
    "Embedding size for embedding matrices."
)
tf.flags.DEFINE_integer("memory_size", 20, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 6")
tf.flags.DEFINE_integer("random_state", 273, "Random state.")
tf.flags.DEFINE_string(
    "data_dir",
    "../babi_tools/dialog-bAbI-tasks/",
    "Directory containing bAbI tasks"
)
FLAGS = tf.flags.FLAGS

print("Started Task:", FLAGS.task_id)

# task data
train, test, oov = load_task(FLAGS.data_dir, FLAGS.task_id)
data = train + test + oov

vocab = sorted(
    reduce(
        lambda x, y: x | y,
        (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)
    )
)
answer_candidates = get_candidates_list(FLAGS.data_dir)
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
answer_idx = dict(
    (candidate, i + 1)
    for i, candidate in enumerate(answer_candidates)
)

max_story_size = max(map(len, (s for s, _, _ in data)))
mean_story_size = int(np.mean([len(s) for s, _, _ in data]))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
query_size = max(map(len, (q for _, q, _ in data)))
memory_size = min(FLAGS.memory_size, max_story_size)
vocab_size = len(word_idx) + 1  # +1 for nil word
answer_vocab_size = len(answer_idx) + 1
sentence_size = max(query_size, sentence_size)  # for the position

print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)

# train/validation/test sets
stories, questions, answers = vectorize_data_dialog(
    data,
    word_idx,
    answer_idx,
    sentence_size,
    memory_size
)

TRAINSET_SIZE = 2
TESTSET_SIZE = len(data) - 2
CROSS_VALIDATION_SPLITTER = ShuffleSplit(
    train_size=TRAINSET_SIZE,
    test_size=TESTSET_SIZE,
    random_state=FLAGS.random_state,
    n_splits=20
)

print("Training Size", TRAINSET_SIZE)
print("Testing Size", TESTSET_SIZE)

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size
optimizer = tf.train.AdamOptimizer(
    learning_rate=FLAGS.learning_rate,
    epsilon=FLAGS.epsilon
)

batches = zip(
    range(0, TRAINSET_SIZE - batch_size, batch_size),
    range(batch_size, TRAINSET_SIZE, batch_size)
)
batches = [(start, end) for start, end in batches]


def train_model(in_model, in_train_sqa, in_test_sqa):
    for t in range(1, FLAGS.epochs+1):
        best_test_accuracy = 0
        s_train, q_train, a_train = in_train_sqa
        s_test, q_test, a_test = in_test_sqa
        train_labels = np.argmax(a_train, axis=1)
        test_labels = np.argmax(a_test, axis=1)
        np.random.shuffle(batches)
        total_cost = 0.0
        for start, end in batches:
            s = s_train[start:end]
            q = q_train[start:end]
            a = a_train[start:end]
            # back-propagating each batch
            cost_t = in_model.batch_fit(s, q, a)
            total_cost += cost_t

        if t % FLAGS.evaluation_interval == 0:
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
            best_test_accuracy = max(best_test_accuracy, test_acc)
    return best_test_accuracy


def main(in_split_number):
    for split_index, (train_indices, test_indices) in enumerate(CROSS_VALIDATION_SPLITTER.split(stories)):
        if split_index != in_split_number:
            continue
        train_s = map(lambda x: stories[x], train_indices)
        train_q = map(lambda x: questions[x], train_indices)
        train_a = map(lambda x: answers[x], train_indices)
        test_s = map(lambda x: stories[x], test_indices)
        test_q = map(lambda x: questions[x], test_indices)
        test_a = map(lambda x: answers[x], test_indices)

        with tf.Session() as sess:
            model = MemN2N(
                batch_size,
                vocab_size,
                sentence_size,
                memory_size,
                FLAGS.embedding_size,
                answer_vocab_size=answer_vocab_size,
                session=sess,
                hops=FLAGS.hops,
                max_grad_norm=FLAGS.max_grad_norm,
                optimizer=optimizer
            )
            best_accuracy_per_epoch = train_model(
                model,
                (train_s, train_q, train_a),
                (test_s, test_q, test_a)
            )
        return best_accuracy_per_epoch


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print ('Usage: dialog_cross_validation.py <split number>')
    print ('{.3f}'.format(main(int(sys.argv[1]))))
