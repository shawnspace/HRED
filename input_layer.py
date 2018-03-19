import tensorflow as tf
import modekeys

def create_input_layer(mode,filenames,num_epochs,batch_size,shuffle_batch,max_sentence_length,max_context_length,max_response_length):
    with tf.name_scope('input_layer') as ns:
        example = read_and_decode(filenames, num_epochs, max_sentence_length,max_context_length,max_response_length)
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * batch_size
        if shuffle_batch:
            batch_example = tf.train.shuffle_batch(example,batch_size=batch_size,
                                                   capacity=capacity,min_after_dequeue=min_after_dequeue)
        else:
            batch_example = tf.train.batch(example,batch_size=batch_size)


        batch_example['context_length'] = tf.squeeze(batch_example['context_length'], 1)

        if mode == modekeys.TRAIN or mode == modekeys.EVAL:
            return batch_example
        elif mode == modekeys.PREDICT:
            return batch_example

def read_and_decode(filenames,num_epochs,max_sentence_length,max_context_length,max_response_length):
    fname_queue = tf.train.string_input_producer(filenames,num_epochs=num_epochs)
    reader = tf.TFRecordReader("my_reader")
    _, serilized_example = reader.read(queue=fname_queue)
    feature_spec = create_feature_spec(max_sentence_length,max_context_length,max_response_length)
    example = tf.parse_single_example(serilized_example, feature_spec)
    example['contexts'] = tf.reshape(example['contexts_flatten'],shape=[max_context_length,max_sentence_length])
    example.pop('contexts_flatten')
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

