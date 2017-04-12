import os

import numpy as np

from data_utils import tokenize


def get_dialogs(f, ignore_api_calls):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_dialogs(f.readlines(), ignore_api_calls=ignore_api_calls)


def load_task(data_dir, task_id, only_supporting=False):
    '''Load the nth task. There are 6 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert 0 < task_id < 7

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'dialog-babi-task{}'.format(task_id)
    train_file = filter(lambda file: s in file and 'trn' in file, files)[0]
    dev_file = filter(lambda file: s in file and 'dev' in file, files)[0]
    test_file = filter(lambda file: s in file and 'tst' in file, files)[0]
    oov_file = filter(lambda file: s in file and 'OOV' in file, files)[0]
    train_data = get_dialogs(train_file, True)
    dev_data = get_dialogs(dev_file, True)
    test_data = get_dialogs(test_file, True)
    oov_data = get_dialogs(oov_file, True)
    return train_data, dev_data, test_data, oov_data


def load_task_for_cv(data_dir, task_id):
    '''Load the nth task. There are 6 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert 0 < task_id < 7

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'dialog-babi-task{}'.format(task_id)
    train_file = filter(lambda file: s in file and 'trn' in file, files)[0]
    dev_file = filter(lambda file: s in file and 'dev' in file, files)[0]
    test_file = filter(lambda file: s in file and 'tst' in file, files)[0]
    oov_file = filter(lambda file: s in file and 'OOV' in file, files)[0]
    files_sorted = sorted([train_file, dev_file, test_file, oov_file])

    all_dialogues = map(lambda x: get_dialogs(x, True), files_sorted)
    return reduce(lambda x, y: x + y, all_dialogues, [])


def parse_dialogs(lines, ignore_api_calls=False):
    data = []
    story = []
    for line in lines:
        line = line.lower().strip()
        if not line:
            continue
        if 'api_call' in line and ignore_api_calls:
            continue
        nid, q_a = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
            data.append([])
        question, answer = q_a.split('\t')
        question = question.rstrip('?')
        question = tokenize(question)
        answer = tokenize(answer)

        # Provide all the substories
        substory = filter(lambda x: x, story)
        data[-1].append((substory, question, answer))
        story.append(question)
        story.append(answer)
    return filter(lambda x: x, data)


def get_candidates_list(data_dir):
    candidates_file = os.path.join(data_dir, 'dialog-babi-candidates.txt')
    with open(candidates_file) as candidates_in:
        return [line.strip().split(' ', 1)[1] for line in candidates_in]


def vectorize_data_dialog(data, word_idx, answer_idx, sentence_size, memory_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(answer_idx) + 1) # 0 is reserved for nil word
        y[answer_idx[' '.join(answer).replace(' \' ', '\'')]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A)
