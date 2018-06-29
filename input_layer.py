import tensorflow as tf
import modekeys

def create_input_layer(filename,hp,mode):
    with tf.name_scope('input_layer') as ns:
        if mode == modekeys.TRAIN or mode == modekeys.EVAL:
            example = read_and_decode([filename], hp.num_epochs, hp.max_sentence_length,hp.max_context_length,hp.max_sentence_length)
            min_after_dequeue = 10000
            capacity = min_after_dequeue + 3 * hp.batch_size
            if hp.shuffle_batch:
                batch_example = tf.train.shuffle_batch(example,batch_size=hp.batch_size,
                                                       capacity=capacity,min_after_dequeue=min_after_dequeue)
            else:
                batch_example = tf.train.batch(example,batch_size=hp.batch_size)


            batch_example['context_length'] = tf.squeeze(batch_example['context_length'], 1)
            return batch_example

        elif mode == modekeys.PREDICT:
            features = {}
            features['contexts'] = tf.placeholder(dtype=tf.int64, shape=[1, hp.max_context_length, hp.max_sentence_length])
            features['context_utterance_length'] = tf.placeholder(dtype=tf.int64, shape=[1,hp.max_context_length])
            features['context_length'] = tf.placeholder(dtype=tf.int64,shape=[1])
            return features

def read_and_decode(filenames,num_epochs,max_sentence_length,max_context_length,max_response_length):
    fname_queue = tf.train.string_input_producer(filenames,num_epochs=num_epochs)
    reader = tf.TFRecordReader("my_reader")
    _, serilized_example = reader.read(queue=fname_queue)
    feature_spec = create_feature_spec(max_sentence_length,max_context_length,max_response_length)
    example = tf.parse_single_example(serilized_example, feature_spec)
    example['contexts'] = tf.reshape(example['contexts_flatten'],shape=[max_context_length,max_sentence_length])
    example.pop('contexts_flatten')
    example['response_mask'] = tf.to_float(example['response_mask'])
    return example

def create_feature_spec(max_sentence_length,max_context_length,max_response_length):
    spec = {}
    spec['contexts_flatten'] = tf.FixedLenFeature(shape=[max_context_length * max_sentence_length],dtype=tf.int64)
    spec['context_utterance_length'] = tf.FixedLenFeature(shape=[max_context_length], dtype=tf.int64)
    spec['context_length'] = tf.FixedLenFeature(shape=[1], dtype=tf.int64)
    spec['response_in'] = tf.FixedLenFeature(shape=[max_response_length], dtype=tf.int64)
    spec['response_out'] = tf.FixedLenFeature(shape=[max_response_length], dtype=tf.int64)
    spec['response_mask'] = tf.FixedLenFeature(shape=[max_response_length], dtype=tf.int64)
    return spec

