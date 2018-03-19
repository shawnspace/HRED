import tensorflow as tf
import os
import csv
import numpy as np

MAX_SENTENCE_LENGTH = 40
MAX_CONTEXT_LENGTH = 10
MAX_RESPONSE_LENGTH = 30

def load_vocabulary(vocab_path): #37446 words
    vocabulary = {}
    with open(vocab_path, 'r') as f:
        for i,l in enumerate(f.readlines()): # unk index = 0 eos index = 1
            vocabulary[l.rstrip('\n')] = i
    print('vocab size {}'.format(len(vocabulary)))
    return vocabulary

def _int64_feature(value):
    # parameter value is a list
    return tf.train.Feature(int64_list = tf.train.Int64List(value = value))

def transform_utterance(utterance,vocab,max_length):
    word_ids = []
    for w in utterance.split(' '):
        if not w == '':
            if w in vocab.keys():
                word_ids.append(vocab[w])
            elif w == '__eou__':
                word_ids.append(1)
            else:
                word_ids.append(0)

    utterance_len = len(word_ids)
    if utterance_len <= max_length:
        word_ids.extend([0]*(max_length - utterance_len))
    else:
        word_ids = word_ids[0:max_length]
        utterance_len = max_length

    return word_ids,utterance_len

def transform_response(response,vocab,max_length):
    word_ids = []
    for w in response.split(' '):
        if not w == '':
            if w in vocab.keys():
                word_ids.append(vocab[w])
            elif w == '__eou__':
                word_ids.append(1)
            else:
                word_ids.append(0)

    response_in = [1] + word_ids[0:len(word_ids) - 1]
    re_in_len = len(response_in)
    if re_in_len <= max_length:
        response_in = response_in + [0]*(max_length - re_in_len)
    else:
        response_in = response_in[0:max_length]
        re_in_len = max_length

    response_out = word_ids[0:len(word_ids) - 1]
    re_out_len = len(response_out)
    response_mask = [1] * re_out_len
    if re_out_len <= max_length -1:
        response_out.extend([1] + [0]*(max_length - re_out_len - 1))
        response_mask.extend([1]+ [0]*(max_length - re_out_len - 1))
    else:
        response_out = response_out[0:max_length - 1]
        response_out.append(1)
        response_mask = response_mask[0:max_length]
        re_out_len = max_length

    return response_in,response_out,response_mask

def create_example(raw_context,raw_response,vocabulary):
    contexts = [c for c in raw_context.split('__eot__') if not c == '']
    context_len = len(contexts)

    context_ids = []
    context_lens = []
    for u in contexts:
        u_ids, u_len = transform_utterance(u,vocabulary,MAX_SENTENCE_LENGTH)
        context_ids.append(u_ids)
        context_lens.append(u_len)

    assert context_len == len(context_ids)

    if context_len <= MAX_CONTEXT_LENGTH:
        a = [0] * MAX_SENTENCE_LENGTH
        context_ids.extend([a] * (MAX_CONTEXT_LENGTH - context_len))
        context_lens.extend([0]*(MAX_CONTEXT_LENGTH - context_len))
    else:
        context_ids = context_ids[context_len - MAX_CONTEXT_LENGTH:context_len]
        context_lens = context_lens[context_len - MAX_CONTEXT_LENGTH:context_len]
        context_len = MAX_CONTEXT_LENGTH
        assert len(context_ids) == context_len

    assert len(context_ids) == len(context_lens)

    response_in, response_out, response_mask = transform_response(raw_response,vocabulary,MAX_RESPONSE_LENGTH)

    context_ids = np.array(context_ids,dtype=np.int64)
    features = {'contexts_flatten':_int64_feature(value = context_ids.flatten()),
                'context_utterance_length':_int64_feature(value=context_lens),
                'context_length':_int64_feature(value=[context_len]),
                'response_in':_int64_feature(response_in),
                'response_out':_int64_feature(response_out),
                'response_mask':_int64_feature(response_mask)}

    example = tf.train.Example(features=tf.train.Features(feature=features))

    return example

def create_dataset(datadir):
    train_file = os.path.join(datadir,'train.csv')
    eval_file = os.path.join(datadir, 'valid.csv')
    test_file = os.path.join(datadir,'test.csv')

    vocabulary = load_vocabulary('data/vocabulary.txt')

    examples = []
    with open(train_file, 'r', newline='') as f:
        reader = csv.reader(f)
        row = reader.__next__()
        for row in reader:
            if int(row[2]) == 1:
                example = create_example(row[0],row[1],vocabulary)
                examples.append(example)

    train_file_path = os.path.join(datadir,'train.tfrecords')
    writer = tf.python_io.TFRecordWriter(train_file_path)
    for e in examples:
        writer.write(e.SerializeToString())
    print('num of train examples {}'.format(len(examples)))

    examples = []
    with open(eval_file,'r', newline='') as f:
        reader = csv.reader(f)
        row = reader.__next__()
        for row in reader:
            example = create_example(row[0],row[1],vocabulary)
            examples.append(example)

    valid_file_path = os.path.join(datadir,'valid.tfrecords')
    writer = tf.python_io.TFRecordWriter(valid_file_path)
    for e in examples:
        writer.write(e.SerializeToString())
    print('num of valid examples {}'.format(len(examples)))

    examples = []
    with open(test_file, 'r', newline='') as f:
        reader = csv.reader(f)
        row = reader.__next__()
        for row in reader:
            example = create_example(row[0], row[1],vocabulary)
            examples.append(example)

    test_file_path = os.path.join(datadir, 'test.tfrecords')
    writer = tf.python_io.TFRecordWriter(test_file_path)
    for e in examples:
        writer.write(e.SerializeToString())
    print('num of test examples {}'.format(len(examples)))

if __name__ == "__main__":
    create_dataset('./data')




