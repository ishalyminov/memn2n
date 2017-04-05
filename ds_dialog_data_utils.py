import os

import numpy as np


def parse_dialogs_ds(lines, ignore_api_calls=False):
    data = []
    story = []
    for line in lines:
        line = line.lower().strip()
        if not line:
            continue
        nid, q_a = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
            data.append([])
        question, answer = q_a.split('\t')
        question = map(int, question.strip('[]').split(', '))
        answer = map(int, answer.strip('[]').split(', '))

        # Provide all the substories
        substory = filter(lambda x: x, story)
        data[-1].append((substory, question, answer))
        story.append(question)
        story.append(answer)
    return filter(lambda x: x, data)


def get_dialogs_ds(f, ignore_api_calls):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_dialogs_ds(
            f.readlines(),
            ignore_api_calls=ignore_api_calls
        )


def load_task_ds(data_dir, task_id):
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
    train_data = get_dialogs_ds(train_file, True)
    dev_data = get_dialogs_ds(dev_file, True)
    test_data = get_dialogs_ds(test_file, True)
    oov_data = get_dialogs_ds(oov_file, True)
    return train_data, dev_data, test_data, oov_data


def ttr_mdp_to_one_hot(in_sentence):
    one_hot_result = []
    for bit_index, bit in enumerate(in_sentence):
        if bit == 1:
            one_hot_result.append(bit_index)
    return one_hot_result


def vectorize_data_dialog_ds(
    data_raw,
    data_ds,
    question_vocabulary_size,
    max_question_sentence_size,
    answer_idx,
    memory_size
):
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

    for (story_raw, query_raw, answer_raw), (story, query, answer) in zip(data_raw, data_ds):
        # sentence is already DS-encoded and of a constant size
        ss = []
        for sentence in story:
            one_hot_sentence = ttr_mdp_to_one_hot(sentence)
            ls = max(0, max_question_sentence_size - len(one_hot_sentence))
            ss.append(one_hot_sentence + [question_vocabulary_size - 1] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * max_question_sentence_size)

        one_hot_query = ttr_mdp_to_one_hot(query)
        lq = max(0, max_question_sentence_size - len(one_hot_query))
        q = one_hot_query + [question_vocabulary_size - 1] * lq

        y = np.zeros(len(answer_idx) + 1) # 0 is reserved for nil word
        y[answer_idx[' '.join(answer_raw).replace(' \' ', '\'')]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A)
